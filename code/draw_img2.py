#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== 与训练一致的导入 ======
from config import get_config
from dataloaders.CC359_dataset_PGIUN_8 import SliceData_CC359


# ----------------------------
# ckpt load
# ----------------------------
def _strip_module_prefix(sd):
    out = OrderedDict()
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[7:]] = v
        else:
            out[k] = v
    return out

def load_weights(model, ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("model_state_dict", raw.get("state_dict", raw))
    if not isinstance(sd, dict):
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
    sd = _strip_module_prefix(sd)
    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            raise RuntimeError(f"Missing keys (not only BN buffers): {missing[:30]} ...")
    if unexpected:
        print("[Warn] Unexpected keys:", unexpected[:20], f"... total={len(unexpected)}")
    print(f"[OK] Loaded: {ckpt_path}")


# ----------------------------
# build model (和你工程一致)
# ----------------------------
def ensure_args_for_config(a):
    defaults = {
        'opts': None,
        'batch_size': None,
        'zip': False,
        'cache_mode': None,
        'resume': '',
        'accumulation_steps': 0,
        'use_checkpoint': False,
        'amp_opt_level': '',
        'tag': None,
        'eval': False,
        'throughput': False,
    }
    for k, v in defaults.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    return a

def build_model(args, device):
    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as Net
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as Net
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as Net
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as Net
    else:
        raise ValueError(args.model)

    cfg = get_config(args)
    model = Net(cfg, patch_size=args.patch_size, num_classes=2, model_type=args.model_type).to(device)
    model.eval()
    return model


# ----------------------------
# find dapso modules (by UC_h/UC_w)
# ----------------------------
def list_dapso_modules(model):
    mods = []
    for name, m in model.named_modules():
        if hasattr(m, "UC_h") and hasattr(m, "UC_w"):
            mods.append((name, m))
    return mods


@torch.no_grad()
def near_diagonal_ratio_hc_only(
    model,
    loader,
    layer_idx=0,
    device="cuda",
    max_batches=30,
    max_samples_per_batch=2048,
):
    """
    只分析 HC：hook UC_h 的 inverse=False 那次（每对调用取第一次）。
    Returns:
      omega: (L,) in [-1,1]  where L=H
      ratio_in:  before UC_h
      ratio_out: after  UC_h  (orthogonal basis applied)
    """
    model.eval().to(device)
    dapso_list = list_dapso_modules(model)
    assert len(dapso_list) > 0, "No DAPSO modules found (need UC_h & UC_w)."
    assert 0 <= layer_idx < len(dapso_list), f"layer_idx out of range: 0..{len(dapso_list)-1}"

    layer_name, dapso = dapso_list[layer_idx]
    uc = dapso.UC_h
    print(f"[Using DAPSO layer] idx={layer_idx} name={layer_name} | Hook UC_h only")

    cov_in_sum = None
    cov_out_sum = None
    count = 0

    # UC_h 在一次 forward 中会被调用两次（inverse=False/True），forward_hook拿不到 kwargs
    # 所以用 call_cnt 只取每对调用的第一次（等价于 inverse=False 那次）。
    call_cnt = 0

    def hook_fn(module, inp, out):
        nonlocal cov_in_sum, cov_out_sum, count, call_cnt

        if (call_cnt % 2) == 1:
            call_cnt += 1
            return
        call_cnt += 1

        zin = inp[0]   # (N,C,L)  N=B*W, L=H
        zout = out     # (N,C,L)

        if not torch.is_complex(zin):
            zin = zin.to(torch.complex64)
        if not torch.is_complex(zout):
            zout = zout.to(torch.complex64)

        N, C, L = zin.shape

        # subsample N to speed up
        if max_samples_per_batch is not None and N > max_samples_per_batch:
            ridx = torch.randperm(N, device=zin.device)[:max_samples_per_batch]
            zin = zin[ridx]
            zout = zout[ridx]
            N = zin.shape[0]

        cov_in = torch.einsum("ncl,ndl->lcd", zin.conj(), zin).detach().cpu()   # (L,C,C)
        cov_out = torch.einsum("ncl,ndl->lcd", zout.conj(), zout).detach().cpu()

        cov_in_sum = cov_in if cov_in_sum is None else (cov_in_sum + cov_in)
        cov_out_sum = cov_out if cov_out_sum is None else (cov_out_sum + cov_out)
        count += N

    h = uc.register_forward_hook(hook_fn)

    nb = 0
    for batch in loader:
        nb += 1
        if nb > max_batches:
            break

        us_image = batch["us_image"].to(device)
        us_mask  = batch["us_mask"].to(device)
        coil_map = batch["coil_map"].to(device)
        _ = model(us_image, us_mask, coil_map)

    h.remove()
    assert cov_in_sum is not None, "Hook didn't capture anything. Check that DAPSO is executed."

    cov_in_mean = cov_in_sum / max(count, 1)
    cov_out_mean = cov_out_sum / max(count, 1)

    def diag_ratio(cov_lcc):
        cov_lcc = cov_lcc.to(torch.complex64)
        total = (cov_lcc.abs() ** 2).sum(dim=(1, 2))  # (L,)
        diag = (torch.diagonal(cov_lcc, dim1=1, dim2=2).abs() ** 2).sum(dim=1)
        return (diag / (total + 1e-12)).numpy()

    ratio_in = diag_ratio(cov_in_mean)
    ratio_out = diag_ratio(cov_out_mean)

    L = cov_in_mean.shape[0]
    omega = np.linspace(-1.0, 1.0, L)
    return omega, ratio_in, ratio_out


def plot_near_diag_hc(omega, rin, rout, out_path, title=None):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(omega, rin, linewidth=1.2, label="Before $U_{C_h}$")
    ax.plot(omega, rout, linewidth=1.2, label="After $U_{C_h}$")
    ax.set_xlabel("Normalized frequency $\\omega$")
    ax.set_ylabel("Diagonal ratio $\\rho(\\omega)$")
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    ap = argparse.ArgumentParser()

    # dataset
    ap.add_argument("--data_dir", type=str, default='/scratch/pf2m24/data/CCP359/Val')
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--mask_type", type=str, default="equispaced")
    ap.add_argument("--resolution", type=int, default=256)

    # model
    ap.add_argument("--model", type=str, default="mamba_unrolled",
                    choices=["mamba_unrolled", "mamba_unet", "swin_unet", "swin_unrolled"])
    ap.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    ap.add_argument("--patch_size", type=int, default=2)
    ap.add_argument("--model_type", type=str, default="dapso")
    ap.add_argument("--ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_ssim_model.pth')

    # analysis
    ap.add_argument("--layer_idx", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=30)
    ap.add_argument("--max_samples_per_batch", type=int, default=2048)

    ap.add_argument("--out_dir", type=str, default="near_diag_hc")
    args = ap.parse_args()
    ensure_args_for_config(args)

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    model = build_model(args, device)
    load_weights(model, args.ckpt, device)

    dataset = SliceData_CC359(
        data_dir=args.data_dir,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        resolution=args.resolution,
        type=args.split,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    omega, rin, rout = near_diagonal_ratio_hc_only(
        model=model,
        loader=loader,
        layer_idx=args.layer_idx,
        device=device,
        max_batches=args.max_batches,
        max_samples_per_batch=args.max_samples_per_batch,
    )

    out_path = os.path.join(args.out_dir, f"“HC channel covariance diagonal ratio before/after Uc")
    plot_near_diag_hc(
        omega, rin, rout,
        out_path=out_path,
        title=f"HC near-diagonal (layer {args.layer_idx})"
    )


if __name__ == "__main__":
    main()

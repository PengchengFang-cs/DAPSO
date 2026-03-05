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
        raise RuntimeError(f"Invalid checkpoint: {ckpt_path}")
    sd = _strip_module_prefix(sd)
    model.load_state_dict(sd, strict=False)
    print(f"[OK] Loaded: {ckpt_path}")


# ----------------------------
# build model (与你工程一致)
# ----------------------------
def ensure_args_for_config(a):
    defaults = {
        'opts': None, 'batch_size': None, 'zip': False, 'cache_mode': None,
        'resume': '', 'accumulation_steps': 0, 'use_checkpoint': False,
        'amp_opt_level': '', 'tag': None, 'eval': False, 'throughput': False,
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
# find DAPSO modules
# ----------------------------
def list_dapso_modules(model):
    mods = []
    for name, m in model.named_modules():
        if hasattr(m, "UC_h") and hasattr(m, "_hc_branch"):  # your DAPSO has both
            mods.append((name, m))
    return mods


# ----------------------------
# linear map estimate per frequency
# ----------------------------
def estimate_M_per_freq(Zin, Zout, ridge=1e-4):
    """
    Zin, Zout: complex tensor [N, C, L]
    For each k: solve Zout_k ≈ Zin_k @ M_k^T  (least squares)
    Return: M: [L, C, C] complex
    """
    # Convert to complex64
    Zin = Zin.to(torch.complex64)
    Zout = Zout.to(torch.complex64)

    N, C, L = Zin.shape
    M = torch.zeros((L, C, C), dtype=torch.complex64, device=Zin.device)

    I = torch.eye(C, dtype=torch.complex64, device=Zin.device)

    for k in range(L):
        X = Zin[:, :, k]   # [N, C]
        Y = Zout[:, :, k]  # [N, C]
        # Solve X @ A ≈ Y  (A is CxC)
        XtX = X.conj().T @ X  # [C, C]
        XtY = X.conj().T @ Y  # [C, C]
        A = torch.linalg.solve(XtX + ridge * I, XtY)  # [C, C]
        M[k] = A
    return M


def offdiag_ratio(M_lcc):
    """
    M_lcc: [L,C,C] complex
    ratio[k] = ||offdiag||_F / ||M||_F
    """
    Mabs2 = (M_lcc.abs() ** 2)
    total = Mabs2.sum(dim=(1, 2))  # [L]
    diag = (torch.diagonal(Mabs2, dim1=1, dim2=2)).sum(dim=1)  # [L]
    off = torch.clamp(total - diag, min=0.0)
    return torch.sqrt(off / (total + 1e-12)).detach().cpu().numpy()


@torch.no_grad()
def compute_offdiag_curves_for_hc_branch(dapso, model, loader, device="cuda",
                                        max_batches=20, max_samples_per_batch=4096,
                                        ridge=1e-4):
    """
    用 forward_pre_hook 抓到“真正进入该 DAPSO 的输入特征 x_in: [B,C,H,W] (C=hidden_dim)”，
    再对 hc_branch 做算子级近对角 proxy：
      - canonical basis: Zin = FFT(xw)
      - learned basis:   Zin = UC_h(FFT(xw))
    并用最小二乘估计每个频率 k 的通道映射矩阵 M(k)，画 off-diagonal mixing ratio。
    """
    dapso.eval()
    model.eval()

    uc = dapso.UC_h

    sum_can = None
    sum_learn = None
    count = 0
    L_ref = None

    # 用 hook 捕获 DAPSO 输入特征
    cache = {"x_in": None}

    def pre_hook(module, inputs):
        # inputs[0] 就是 forward(x) 的 x: [B,C,H,W]
        cache["x_in"] = inputs[0].detach()

    h = dapso.register_forward_pre_hook(pre_hook)

    nb = 0
    for batch in loader:
        nb += 1
        if nb > max_batches:
            break

        us_image = batch["us_image"].to(device)
        us_mask  = batch["us_mask"].to(device)
        coil_map = batch["coil_map"].to(device)

        cache["x_in"] = None
        _ = model(us_image, us_mask, coil_map)   # 触发 hook，拿到 x_in

        x = cache["x_in"]
        if x is None:
            # 说明这轮 forward 没走到该 dapso（可能 layer_idx 选错或某分支没启用）
            continue

        x = x.to(device)
        B, C, H, W = x.shape

        # --- 输入侧：canonical vs learned basis ---
        xw = x.permute(0, 3, 1, 2).reshape(B * W, C, H)  # [BW, C, H]
        if max_samples_per_batch is not None and xw.shape[0] > max_samples_per_batch:
            ridx = torch.randperm(xw.shape[0], device=device)[:max_samples_per_batch]
            xw = xw[ridx]

        Zin_can = torch.fft.fft(xw, dim=-1)          # [N,C,H] complex
        Zin_learn = uc(Zin_can, inverse=False)       # [N,C,H] complex

        # --- 输出侧：用 hc_branch 的输出（更贴近“算子”）---
        y = dapso._hc_branch(x)                      # [B,C,H,W] real
        yw = y.permute(0, 3, 1, 2).reshape(B * W, C, H)
        if max_samples_per_batch is not None and yw.shape[0] > max_samples_per_batch:
            ridx = torch.randperm(yw.shape[0], device=device)[:max_samples_per_batch]
            yw = yw[ridx]

        Zout_can = torch.fft.fft(yw, dim=-1)
        Zout_learn = uc(Zout_can, inverse=False)

        # --- 估计每个频率 k 的通道映射矩阵 M(k) 并计算 offdiag ratio ---
        M_can = estimate_M_per_freq(Zin_can, Zout_can, ridge=ridge)        # [H,C,C]
        M_learn = estimate_M_per_freq(Zin_learn, Zout_learn, ridge=ridge)  # [H,C,C]

        r_can = offdiag_ratio(M_can)     # [H]
        r_learn = offdiag_ratio(M_learn) # [H]

        if sum_can is None:
            L_ref = r_can.shape[0]
            sum_can = np.zeros((L_ref,), dtype=np.float64)
            sum_learn = np.zeros((L_ref,), dtype=np.float64)

        sum_can += r_can
        sum_learn += r_learn
        count += 1

    h.remove()

    if count == 0:
        raise RuntimeError("No valid samples captured from the selected DAPSO module. "
                           "Check layer_idx / whether this DAPSO is actually executed.")

    omega = np.linspace(-1.0, 1.0, L_ref)
    return omega, (sum_can / count), (sum_learn / count)



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

    # which dapso
    ap.add_argument("--layer_idx", type=int, default=7)

    # compute
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=100)
    ap.add_argument("--max_samples_per_batch", type=int, default=2048)
    ap.add_argument("--ridge", type=float, default=1e-4)

    ap.add_argument("--out", type=str, default=f"verify_theorem_hc_offdiag_7.png")

    args = ap.parse_args()
    ensure_args_for_config(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    model = build_model(args, device)
    load_weights(model, args.ckpt, device)

    # find dapso block
    dapso_list = list_dapso_modules(model)
    assert len(dapso_list) > 0, "No DAPSO modules found."
    name, dapso = dapso_list[args.layer_idx]
    print(f"[Using DAPSO] idx={args.layer_idx} name={name}")

    # data
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

    omega, r_can, r_learn = compute_offdiag_curves_for_hc_branch(
        dapso=dapso,
        model=model,          # <<< 新增
        loader=loader,
        device=device,
        max_batches=args.max_batches,
        max_samples_per_batch=args.max_samples_per_batch,
        ridge=args.ridge,
    )


    # plot
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(omega, r_can, linewidth=1.2, label="Canonical channel basis")
    ax.plot(omega, r_learn, linewidth=1.2, label="Learned $U_{C_h}$ basis")
    ax.set_xlabel("Normalized frequency $\\omega$")
    ax.set_ylabel("Off-diagonal mixing ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=250)
    plt.close(fig)
    print(f"[Saved] {args.out}")


if __name__ == "__main__":
    main()

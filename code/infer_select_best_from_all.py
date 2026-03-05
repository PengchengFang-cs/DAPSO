#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import heapq
from collections import OrderedDict

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity, peak_signal_noise_ratio

# ====== 与训练一致的导入 ======
from config import get_config
from dataloaders.CC359_dataset_PGIUN_8 import SliceData_CC359


# ----------------------------
# Metrics
# ----------------------------
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-12)

def psnr(gt, pred):
    dr = max(gt.max() - gt.min(), 1e-12)
    return peak_signal_noise_ratio(gt, pred, data_range=dr)

def ssim(gt, pred):
    dr = max(gt.max() - gt.min(), 1e-12)
    return structural_similarity(gt, pred, data_range=dr)


# ----------------------------
# CKPT loading
# ----------------------------
def _strip_module_prefix(state_dict):
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

def load_weights(model, ckpt_path, device):
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("model_state_dict", raw.get("state_dict", raw))
    if not isinstance(sd, dict):
        raise RuntimeError("Invalid checkpoint format, no state dict found.")
    sd = _strip_module_prefix(sd)

    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    # 仅允许 BN buffer 类缺失
    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            raise RuntimeError(f"Missing keys (not only BN buffers): {missing[:30]} ... total={len(missing)}")

    if unexpected:
        # unexpected 通常不应该有；你想更严格可以直接 raise
        print("[Warn] Unexpected keys:", unexpected[:30], f"... total={len(unexpected)}")

    print(f"[OK] Loaded: {ckpt_path}")


# ----------------------------
# Model builder (与你训练逻辑一致)
# ----------------------------
def build_model(args, device, model_type=None):
    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as Net
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as Net
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as Net
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as Net
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    config = get_config(args)
    model = Net(config, patch_size=args.patch_size, num_classes=2, model_type=model_type).to(device)
    model.eval()
    return model


# ----------------------------
# Helpers
# ----------------------------
def ensure_args_for_config(ns):
    # 补齐 config.update_config 可能访问字段
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
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns

def to_4d(x: torch.Tensor) -> torch.Tensor:
    """[H,W] / [C,H,W] / [B,C,H,W] -> [1,C,H,W]"""
    if x.ndim == 2:
        return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim == 4:
        return x
    raise ValueError(f"Unexpected shape: {tuple(x.shape)}")

def unwrap_output(out):
    if isinstance(out, (list, tuple)):
        return out[0]
    return out

def tensor_to_mag(x):
    """
    input: torch.Tensor or numpy.ndarray
    output: 2D numpy magnitude [H,W]
    supports: [B,2,H,W],[2,H,W],[B,1,H,W],[1,H,W],[H,W]
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 3:
        C, H, W = x.shape
        if C == 2:
            mag = torch.sqrt(x[0] ** 2 + x[1] ** 2)
            return mag.cpu().numpy()
        elif C == 1:
            return x[0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected channels: {C}")
    elif x.ndim == 2:
        return x.cpu().numpy()
    else:
        raise ValueError(f"Unexpected ndim={x.ndim}, shape={tuple(x.shape)}")

def save_recon_grid_png(us_mag, gt_mag, recons, titles, save_path):
    """
    1行N列：Under, GT, Ours, M1, M2, M3（共6张）
    recons: list of 2D arrays (ours + others)
    titles: list of strings for each recon
    """
    # images order: Under, GT, (recons...)
    imgs = [us_mag, gt_mag] + recons
    names = ["Under", "GT"] + titles

    vmax = max([im.max() for im in imgs])  # 你要求：vmax 不做额外修改

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, im, name in zip(axes, imgs, names):
        ax.imshow(im, cmap="gray", vmin=0, vmax=vmax)
        ax.set_title(name)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

def save_error_grid_png(gt_mag, recons, titles, save_path):
    """
    1行N列：Err_Ours, Err_M1, Err_M2, Err_M3
    """
    errs = [np.abs(r - gt_mag) for r in recons]
    n = len(errs)
    evmax = max([e.max() for e in errs]) if n > 0 else 1.0

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes]

    for ax, e, name in zip(axes, errs, titles):
        ax.imshow(e, cmap="magma", vmin=0, vmax=evmax)
        ax.set_title(f"Err_{name}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_models_on_sample(models, us_image, us_mask, coil_map, device):
    """
    models: list[torch.nn.Module]
    return: list of recon_mag (2D numpy) corresponding to each model
    """
    recons_mag = []
    with torch.no_grad():
        for m in models:
            out = unwrap_output(m(us_image, us_mask, coil_map))
            # out expected [B,2,H,W] or [2,H,W]
            recon_mag = tensor_to_mag(out)
            recons_mag.append(recon_mag)
    return recons_mag


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument("--data_dir", type=str, default='/scratch/pf2m24/data/CCP359/Val')
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--acceleration", type=int, default=8)
    parser.add_argument("--mask_type", type=str, default="equispaced")
    parser.add_argument("--resolution", type=int, default=256)

    # output
    parser.add_argument("--save_dir", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/results/cc359_8x')
    parser.add_argument("--topk", type=int, default=10, help="选几张（你说只要1张就保持默认）")
    parser.add_argument("--save_npy", action="store_true", help="是否额外保存np y(不建议当作看图的主输出)")
    parser.add_argument("--eps", type=float, default=1e-9, help="严格大于的数值容忍(默认接近0)")

    # ours
    parser.add_argument("--ours_model", type=str, default="mamba_unrolled",
                        choices=["mamba_unrolled", "mamba_unet", "swin_unet", "swin_unrolled"])
    parser.add_argument("--ours_cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    parser.add_argument("--ours_patch_size", type=int, default=2)
    parser.add_argument("--ours_ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')

    # other 3 models
    # 不填 model/cfg/patch_size 就默认跟 ours 一样（只换 ckpt）
    parser.add_argument("--m1_ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_deformationfno_1130/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument("--m1_model", type=str, default=None,
                        choices=[None, "mamba_unrolled", "mamba_unet", "swin_unet", "swin_unrolled"])
    parser.add_argument("--m1_cfg", type=str, default=None)
    parser.add_argument("--m1_patch_size", type=int, default=None)

    parser.add_argument("--m2_ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_fsel_1130/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument("--m2_model", type=str, default=None,
                        choices=[None, "mamba_unrolled", "mamba_unet", "swin_unet", "swin_unrolled"])
    parser.add_argument("--m2_cfg", type=str, default=None)
    parser.add_argument("--m2_patch_size", type=int, default=None)

    parser.add_argument("--m3_ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_spect_1130/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument("--m3_model", type=str, default=None,
                        choices=[None, "mamba_unrolled", "mamba_unet", "swin_unet", "swin_unrolled"])
    parser.add_argument("--m3_cfg", type=str, default=None)
    parser.add_argument("--m3_patch_size", type=int, default=None)

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    # build 4 model args
    def make_ns(model, cfg, patch_size):
        ns = argparse.Namespace()
        ns.model = model
        ns.cfg = cfg
        ns.patch_size = patch_size
        ensure_args_for_config(ns)
        return ns

    ours_ns = make_ns(args.ours_model, args.ours_cfg, args.ours_patch_size)

    def inherit_or(v, fallback):
        return fallback if v is None else v

    m1_ns = make_ns(
        inherit_or(args.m1_model, args.ours_model),
        inherit_or(args.m1_cfg, args.ours_cfg),
        inherit_or(args.m1_patch_size, args.ours_patch_size),
    )
    m2_ns = make_ns(
        inherit_or(args.m2_model, args.ours_model),
        inherit_or(args.m2_cfg, args.ours_cfg),
        inherit_or(args.m2_patch_size, args.ours_patch_size),
    )
    m3_ns = make_ns(
        inherit_or(args.m3_model, args.ours_model),
        inherit_or(args.m3_cfg, args.ours_cfg),
        inherit_or(args.m3_patch_size, args.ours_patch_size),
    )

    # build + load weights
    ours_model = build_model(ours_ns, device, model_type='dapso')
    load_weights(ours_model, args.ours_ckpt, device)

    m1_model = build_model(m1_ns, device, model_type='deformfno')
    load_weights(m1_model, args.m1_ckpt, device)

    m2_model = build_model(m2_ns, device, model_type='fsel')
    load_weights(m2_model, args.m2_ckpt, device)

    m3_model = build_model(m3_ns, device, model_type='spectformer')
    load_weights(m3_model, args.m3_ckpt, device)

    models = [ours_model, m1_model, m2_model, m3_model]
    model_names = [
        f"Ours({ours_ns.model})",
        f"M1({m1_ns.model})",
        f"M2({m2_ns.model})",
        f"M3({m3_ns.model})",
    ]

    # dataset
    dataset = SliceData_CC359(
        data_dir=args.data_dir,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        resolution=args.resolution,
        type=args.split,
    )
    N = len(dataset)
    print(f"Dataset size: {N}")

    # tqdm optional
    try:
        from tqdm import tqdm
        it = tqdm(range(N), desc="Scanning (strict PSNR+SSIM)")
    except Exception:
        it = range(N)

    # heap for topk strict candidates (keep best scores)
    K = max(1, int(args.topk))
    heap = []
    strict_count = 0

    # We rank by mean delta PSNR (primary) + mean delta SSIM (secondary),
    # while also keeping min margins as tie-breakers.
    # score tuple (mean_dpsnr, mean_dssim, min_dpsnr, min_dssim, ours_psnr)
    eps = float(args.eps)

    with torch.no_grad():
        for idx in it:
            sample = dataset[idx]

            us_image = to_4d(sample["us_image"]).to(device)   # [1,2,H,W]
            us_mask  = to_4d(sample["us_mask"]).to(device)    # [1,1,H,W]
            coil_map = to_4d(sample["coil_map"]).to(device)   # [1,1,H,W]
            fs_image = sample["fs_image"]                     # [2,H,W] (numpy/torch)

            us_mag = tensor_to_mag(us_image)
            gt_mag = tensor_to_mag(fs_image)

            # run 4 models
            recons_mag = run_models_on_sample(models, us_image, us_mask, coil_map, device)

            # metrics for each model
            psnrs = [psnr(gt_mag, r) for r in recons_mag]
            ssims = [ssim(gt_mag, r) for r in recons_mag]

            ours_psnr, ours_ssim = psnrs[0], ssims[0]
            other_psnrs = psnrs[1:]
            other_ssims = ssims[1:]

            # strict filter: ours must beat ALL 3 on PSNR and SSIM
            ok_psnr = all((ours_psnr > p + eps) for p in other_psnrs)
            ok_ssim = all((ours_ssim > s + eps) for s in other_ssims)

            if not (ok_psnr and ok_ssim):
                continue

            strict_count += 1

            dpsnr = [ours_psnr - p for p in other_psnrs]
            dssim = [ours_ssim - s for s in other_ssims]
            mean_dpsnr = float(np.mean(dpsnr))
            mean_dssim = float(np.mean(dssim))
            min_dpsnr  = float(np.min(dpsnr))
            min_dssim  = float(np.min(dssim))

            score = (mean_dpsnr, mean_dssim, min_dpsnr, min_dssim, float(ours_psnr))

            record = {
                "idx": int(idx),
                "score": score,
                "psnr": [float(x) for x in psnrs],
                "ssim": [float(x) for x in ssims],
                "nmse": [float(nmse(gt_mag, r)) for r in recons_mag],
            }

            item = (score, idx, record)
            if len(heap) < K:
                heapq.heappush(heap, item)
            else:
                # keep best K (largest score)
                if score > heap[0][0]:
                    heapq.heapreplace(heap, item)

    best = sorted(heap, key=lambda x: x[0], reverse=True)

    print(f"\nStrict candidates found: {strict_count}")
    if len(best) == 0:
        print("[ERROR] 没有任何样本能同时在 PSNR 和 SSIM 上严格超过另外三个模型。")
        print("你可以：")
        print("  1) 改成 topk>1 看看有没有接近的；或")
        print("  2) 把 eps 设小（默认已很小）；或")
        print("  3) 先只要求 PSNR 过，再肉眼看 SSIM；但你说要严格，这里就不自动放宽。")
        return

    # save summary csv/json and save images for selected
    out_root = os.path.join(args.save_dir, f"STRICT_psnr_ssim_top{len(best)}")
    os.makedirs(out_root, exist_ok=True)

    # summary json
    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump([b[2] for b in best], f, indent=2)
    print("Saved:", summary_path)

    # For each selected idx, re-run and save PNGs (so we don't keep arrays in RAM)
    with torch.no_grad():
        for rank, (score, idx, record) in enumerate(best, start=1):
            sample = dataset[idx]

            us_image = to_4d(sample["us_image"]).to(device)
            us_mask  = to_4d(sample["us_mask"]).to(device)
            coil_map = to_4d(sample["coil_map"]).to(device)
            fs_image = sample["fs_image"]

            us_mag = tensor_to_mag(us_image)
            gt_mag = tensor_to_mag(fs_image)

            recons_mag = run_models_on_sample(models, us_image, us_mask, coil_map, device)

            # output folder
            mean_dpsnr, mean_dssim, min_dpsnr, min_dssim, ours_psnr = score
            tag = (f"rank{rank:02d}_idx{idx:06d}"
                   f"_mdPSNR{mean_dpsnr:.2f}_mdSSIM{mean_dssim:.4f}"
                   f"_minPSNR{min_dpsnr:.2f}_minSSIM{min_dssim:.4f}")
            odir = os.path.join(out_root, tag)
            os.makedirs(odir, exist_ok=True)

            # save grids
            save_recon_grid_png(
                us_mag=us_mag,
                gt_mag=gt_mag,
                recons=[recons_mag[0], recons_mag[1], recons_mag[2], recons_mag[3]],
                titles=model_names,
                save_path=os.path.join(odir, "compare_recons.png"),
            )
            save_error_grid_png(
                gt_mag=gt_mag,
                recons=[recons_mag[0], recons_mag[1], recons_mag[2], recons_mag[3]],
                titles=model_names,
                save_path=os.path.join(odir, "compare_errors.png"),
            )

            # save individual PNGs (便于你直接拿来排版)
            vmax = max([us_mag.max(), gt_mag.max()] + [r.max() for r in recons_mag])
            plt.imsave(os.path.join(odir, "under.png"),  us_mag, cmap="gray", vmin=0, vmax=vmax)
            plt.imsave(os.path.join(odir, "gt.png"),     gt_mag, cmap="gray", vmin=0, vmax=vmax)

            plt.imsave(os.path.join(odir, "ours.png"),   recons_mag[0], cmap="gray", vmin=0, vmax=vmax)
            plt.imsave(os.path.join(odir, "m1.png"),     recons_mag[1], cmap="gray", vmin=0, vmax=vmax)
            plt.imsave(os.path.join(odir, "m2.png"),     recons_mag[2], cmap="gray", vmin=0, vmax=vmax)
            plt.imsave(os.path.join(odir, "m3.png"),     recons_mag[3], cmap="gray", vmin=0, vmax=vmax)

            plt.imsave(os.path.join(odir, "err_ours.png"), np.abs(recons_mag[0] - gt_mag), cmap="magma")
            plt.imsave(os.path.join(odir, "err_m1.png"),   np.abs(recons_mag[1] - gt_mag), cmap="magma")
            plt.imsave(os.path.join(odir, "err_m2.png"),   np.abs(recons_mag[2] - gt_mag), cmap="magma")
            plt.imsave(os.path.join(odir, "err_m3.png"),   np.abs(recons_mag[3] - gt_mag), cmap="magma")

            # save metrics text
            metrics_txt = os.path.join(odir, "metrics.txt")
            with open(metrics_txt, "w") as f:
                f.write(f"idx: {idx}\n")
                f.write(f"score(mean_dpsnr, mean_dssim, min_dpsnr, min_dssim, ours_psnr): {score}\n\n")
                for name, p, s in zip(model_names, record["psnr"], record["ssim"]):
                    f.write(f"{name:18s}  PSNR={p:.6f}  SSIM={s:.6f}\n")

            # optional NPY (默认不存，你想保留原始浮点就打开 --save_npy)
            if args.save_npy:
                np.save(os.path.join(odir, "under.npy"), us_mag)
                np.save(os.path.join(odir, "gt.npy"), gt_mag)
                np.save(os.path.join(odir, "ours.npy"), recons_mag[0])
                np.save(os.path.join(odir, "m1.npy"), recons_mag[1])
                np.save(os.path.join(odir, "m2.npy"), recons_mag[2])
                np.save(os.path.join(odir, "m3.npy"), recons_mag[3])

            print(f"[Saved] {odir}")

    print("\nDone. Outputs in:", os.path.abspath(out_root))


if __name__ == "__main__":
    main()

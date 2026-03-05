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
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
    sd = _strip_module_prefix(sd)

    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            raise RuntimeError(f"Missing keys (not only BN buffers) in {ckpt_path}: {missing[:30]} ...")
    if unexpected:
        print(f"[Warn] Unexpected keys total={len(unexpected)}")

    print(f"[OK] Loaded: {ckpt_path}")


# ----------------------------
# Model builder
# ----------------------------
def ensure_args_for_config(ns):
    defaults = {
        'opts': None, 'batch_size': None, 'zip': False, 'cache_mode': None,
        'resume': '', 'accumulation_steps': 0, 'use_checkpoint': False,
        'amp_opt_level': '', 'tag': None, 'eval': False, 'throughput': False,
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns

def build_model(base_args, device, model_type=None):
    if base_args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as Net
    elif base_args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as Net
    elif base_args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as Net
        if base_args.cfg is None:
            base_args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif base_args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as Net
        if base_args.cfg is None:
            base_args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    else:
        raise ValueError(f"Unknown model type: {base_args.model}")

    config = get_config(base_args)
    model = Net(config, patch_size=base_args.patch_size, num_classes=2, model_type=model_type).to(device)
    model.eval()
    return model


# ----------------------------
# Tensor utils
# ----------------------------
def to_tensor(x):
    if torch.is_tensor(x): return x
    return torch.from_numpy(np.asarray(x))

def to_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2: return x.unsqueeze(0).unsqueeze(0)
    if x.ndim == 3: return x.unsqueeze(0)
    if x.ndim == 4: return x
    raise ValueError(f"Unexpected ndim={x.ndim}")

def unwrap_output(out):
    if isinstance(out, (tuple, list)): return out[0]
    return out

def to_complex2d(arr) -> np.ndarray:
    if torch.is_tensor(arr):
        arr = arr.detach().cpu()
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        arr = arr.numpy()
    else:
        arr = np.asarray(arr)

    if arr.ndim == 3 and arr.shape[0] == 2:
        return (arr[0] + 1j * arr[1]).astype(np.complex64)
    if arr.ndim == 2:
        return arr.astype(np.complex64)
    raise ValueError(f"Expected (2,H,W) or (H,W), got {arr.shape}")


# ----------------------------
# Radial bin precompute
# ----------------------------
class RadialBinner:
    def __init__(self, H: int, W: int, n_bins: int = 256, eps: float = 1e-12):
        self.H, self.W, self.n_bins, self.eps = H, W, n_bins, eps
        cy, cx = H // 2, W // 2
        y, x = np.indices((H, W))
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_max = min(H, W) / 2.0
        r_norm = np.clip(r / (r_max + eps), 0.0, 1.0)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(r_norm.ravel(), edges) - 1
        idx = np.clip(idx, 0, n_bins - 1).astype(np.int32)
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        self.idx = idx
        self.counts = counts
        self.r = 0.5 * (edges[:-1] + edges[1:])

    def profile_from_power(self, power2d: np.ndarray) -> np.ndarray:
        s = np.bincount(self.idx, weights=power2d.ravel(), minlength=self.n_bins).astype(np.float64)
        prof = s / np.maximum(self.counts, 1.0)
        prof = prof / (prof.sum() + self.eps)
        return prof

def error_spectrum_profile(pred_c: np.ndarray, gt_c: np.ndarray, binner: RadialBinner) -> np.ndarray:
    e = pred_c - gt_c
    E = np.fft.fftshift(np.fft.fft2(e))
    power = (np.abs(E) ** 2).astype(np.float64)
    return binner.profile_from_power(power)


# ----------------------------
# Plotting Style Config (Publication Ready)
# ----------------------------
def set_publication_style():
    """设置适合论文发表的绘图风格：大字体、清晰线条"""
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',          # 衬线字体，配合 LaTeX
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'], 
        'font.size': 14,                 # 全局基础字号
        
        # 坐标轴
        'axes.labelsize': 16,            # x, y 轴标签字号
        'axes.titlesize': 16,            # 标题字号
        'axes.linewidth': 1.5,           # 坐标轴边框粗细
        
        # 刻度
        'xtick.labelsize': 14,           # x 刻度数字大小
        'ytick.labelsize': 14,           # y 刻度数字大小
        'xtick.major.width': 1.5,        # 刻度线粗细
        'ytick.major.width': 1.5,
        
        # 图例
        'legend.fontsize': 13,           # 图例字号
        'legend.frameon': False,         # 图例去掉边框（可选，显得更简洁）或保留
        
        # 线条
        'lines.linewidth': 2.0,          # 默认线条粗细
        'grid.linewidth': 1.0,
    })


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # ... args definition ...
    ap.add_argument("--data_dir", type=str, default='/scratch/pf2m24/data/CCP359/Val')
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--mask_type", type=str, default="equispaced")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--model", type=str, default="mamba_unrolled")
    ap.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    ap.add_argument("--patch_size", type=int, default=2)

    # 6 models
    ap.add_argument("--ckpts", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' \
    '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_gfnet_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' \
    '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_ffc_1201/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' \
    '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_deformationfno_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' \
    '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_fsel_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' \
    '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_spect_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth')
    ap.add_argument("--names", type=str, default="dapso,gfnet,ffc,deformfno,fsel,spectformer")
    ap.add_argument("--model_types", type=str, default="dapso,gfnet,ffc,deformfno,fsel,spectformer")

    ap.add_argument("--n_bins", type=int, default=256)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--logy", action="store_true")
    # 默认后缀改为 pdf，因为 pdf 是矢量图，插入 latex 最清晰
    ap.add_argument("--out", type=str, default="radial_error_spectrum.pdf")

    args = ap.parse_args()

    # 应用画图风格配置
    set_publication_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Device:", device)

    ckpts = [x.strip() for x in args.ckpts.split(",") if x.strip()]
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    mtypes = [x.strip() for x in args.model_types.split(",") if x.strip()]

    if len(ckpts) != 6 or len(names) != 6 or len(mtypes) != 6:
        raise ValueError("Must provide exactly 6 items for ckpts, names, and model_types")

    mtypes = [None if mt.lower() in ("none", "null", "") else mt for mt in mtypes]

    base = argparse.Namespace(model=args.model, cfg=args.cfg, patch_size=args.patch_size)
    base = ensure_args_for_config(base)

    models = []
    for i in range(6):
        m = build_model(base, device, model_type=mtypes[i])
        load_weights(m, ckpts[i], device)
        m.eval()
        models.append(m)
        print(f"[Loaded] {names[i]}")

    dataset = SliceData_CC359(
        data_dir=args.data_dir, acceleration=args.acceleration,
        mask_type=args.mask_type, resolution=args.resolution, type=args.split,
    )
    N = len(dataset)
    use_N = N if args.max_items is None else min(N, int(args.max_items))
    print(f"Dataset size: {N} | Using: {use_N}")

    try:
        from tqdm import tqdm
        it = tqdm(range(use_N), desc="Computing curves")
    except Exception:
        it = range(use_N)

    binner = None
    sum_prof = [None] * 6
    count = 0

    with torch.no_grad():
        for idx in it:
            sample = dataset[idx]
            us_image = to_4d(to_tensor(sample["us_image"])).to(device)
            us_mask  = to_4d(to_tensor(sample["us_mask"])).to(device)
            coil_map = to_4d(to_tensor(sample["coil_map"])).to(device)
            gt_c = to_complex2d(sample["fs_image"])

            if binner is None:
                H, W = gt_c.shape
                binner = RadialBinner(H, W, n_bins=args.n_bins)
                for i in range(6):
                    sum_prof[i] = np.zeros((args.n_bins,), dtype=np.float64)

            for i, m in enumerate(models):
                out = unwrap_output(m(us_image, us_mask, coil_map))
                pred_c = to_complex2d(out)
                prof = error_spectrum_profile(pred_c, gt_c, binner)
                sum_prof[i] += prof
            count += 1

    # ... (前面的代码保持不变) ...

    # ----------------------------
    # Plot Logic Updated (Zoomed in 0-0.6)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    r = binner.r
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    
    # 颜色循环
    colors = plt.get_cmap("tab10").colors 

    MAIN_I = 0 # 假设第 0 个是你的主方法

    for i in range(6):
        mean_prof = sum_prof[i] / max(count, 1)

        if i == MAIN_I:
            lw = 3.0   # 主方法线条加粗
            alpha = 1.0
            z = 10
        else:
            lw = 2.0   # 其他方法
            alpha = 0.9
            z = 5

        ax.plot(
            r, mean_prof,
            linewidth=lw,
            alpha=alpha,
            linestyle=linestyles[i % len(linestyles)],
            label=names[i],
            zorder=z,
        )

    # ====== 核心修改在这里 ======
    # 强制截断 X 轴，只显示 0 到 0.6 的区域
    ax.set_xlim(0, 0.6)
    # ==========================

    ax.set_xlabel("Normalized Frequency Radius")
    ax.set_ylabel("Normalized Error Spectrum Energy")
    
    if args.logy:
        ax.set_yscale("log")
    
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    ax.legend(
        loc="upper right",
        framealpha=0.95,
        borderpad=0.6,
        labelspacing=0.4,
        handlelength=2.5,
    )

    fig.tight_layout()

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    
    # 依然建议存为 pdf
    if not out_path.endswith(".pdf"):
        out_path = os.path.splitext(out_path)[0] + ".pdf"
        
    fig.savefig(out_path, dpi=300, bbox_inches='tight')

    plt.close(fig)
    print(f"[Saved] {out_path} (Range: 0-0.6)")

if __name__ == "__main__":
    main()
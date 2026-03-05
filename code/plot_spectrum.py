#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import OrderedDict
import numpy as np
import torch

# 必须在导入 pyplot 之前设置
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

# ====== 根据你的环境导入 ======
# 请确保这些文件在你的 PYTHONPATH 或当前目录下
from config import get_config
from dataloaders.CC359_dataset_PGIUN_8 import SliceData_CC359

# ----------------------------
# 绘图风格配置 (针对 LaTeX 优化)
# ----------------------------
def set_publication_style():
    """设置大字体、粗线条，适合论文插图"""
    plt.rcParams.update({
        # 字体
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 16,            # 全局基础字号加大
        
        # 坐标轴
        'axes.labelsize': 18,       # 轴标题字号
        'axes.titlesize': 18,
        'axes.linewidth': 2.0,      # 坐标轴边框加粗
        
        # 刻度
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        
        # 图例
        'legend.fontsize': 14,
        'legend.frameon': False,    # 去掉图例边框，显得更干净
        
        # 线条
        'lines.linewidth': 2.5,     # 线条加粗
        'grid.linewidth': 1.0,
        'grid.alpha': 0.4,
    })

# ----------------------------
# Checkpoint Loading
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
    if not os.path.exists(ckpt_path):
        print(f"[Error] Checkpoint not found: {ckpt_path}")
        return
        
    raw = torch.load(ckpt_path, map_location=device)
    sd = raw.get("model_state_dict", raw.get("state_dict", raw))
    
    if not isinstance(sd, dict):
        raise RuntimeError(f"Invalid checkpoint format: {ckpt_path}")
    sd = _strip_module_prefix(sd)

    incompatible = model.load_state_dict(sd, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    
    if missing:
        # 过滤掉 BatchNorm 的统计量缺失警告
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            print(f"[Warn] Missing keys in {os.path.basename(ckpt_path)}: {missing[:5]} ...")
    
    print(f"[OK] Loaded: {os.path.basename(ckpt_path)}")

# ----------------------------
# Model Builder
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
    # 根据你的工程结构动态导入
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
    # 这里的 patch_size, num_classes 等参数需与你训练时一致
    model = Net(config, patch_size=base_args.patch_size, num_classes=2, model_type=model_type).to(device)
    model.eval()
    return model

# ----------------------------
# Tensor Utils
# ----------------------------
def to_tensor(x):
    if torch.is_tensor(x): return x
    return torch.from_numpy(np.asarray(x))

def to_4d(x: torch.Tensor) -> torch.Tensor:
    # 统一转为 [B, C, H, W] 或 [1, 1, H, W]
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
# Radial Binner (计算径向误差谱)
# ----------------------------
class RadialBinner:
    def __init__(self, H: int, W: int, n_bins: int = 256, eps: float = 1e-12):
        self.H, self.W, self.n_bins, self.eps = H, W, n_bins, eps
        cy, cx = H // 2, W // 2
        y, x = np.indices((H, W))
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r_max = min(H, W) / 2.0
        # 归一化半径 0.0 ~ 1.0
        r_norm = np.clip(r / (r_max + eps), 0.0, 1.0)
        
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.digitize(r_norm.ravel(), edges) - 1
        idx = np.clip(idx, 0, n_bins - 1).astype(np.int32)
        
        counts = np.bincount(idx, minlength=n_bins).astype(np.float64)
        self.idx = idx
        self.counts = counts
        # Bin centers
        self.r = 0.5 * (edges[:-1] + edges[1:])

    def profile_from_power(self, power2d: np.ndarray) -> np.ndarray:
        s = np.bincount(self.idx, weights=power2d.ravel(), minlength=self.n_bins).astype(np.float64)
        prof = s / np.maximum(self.counts, 1.0)
        # 归一化，使总和为 1 (Probability Density 概念) 或者保持能量 (视需求而定)
        # 这里保持你原本的逻辑：归一化
        prof = prof / (prof.sum() + self.eps)
        return prof

def error_spectrum_profile(pred_c: np.ndarray, gt_c: np.ndarray, binner: RadialBinner) -> np.ndarray:
    e = pred_c - gt_c
    # FFTShift 将低频移到中心
    E = np.fft.fftshift(np.fft.fft2(e))
    power = (np.abs(E) ** 2).astype(np.float64)
    return binner.profile_from_power(power)

# ----------------------------
# Main Logic
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    
    # --- Dataset & Model Args ---
    ap.add_argument("--data_dir", type=str, default='/scratch/pf2m24/data/CCP359/Val')
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--mask_type", type=str, default="equispaced")
    ap.add_argument("--resolution", type=int, default=256)
    
    ap.add_argument("--model", type=str, default="mamba_unrolled")
    ap.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    ap.add_argument("--patch_size", type=int, default=2)

    # --- 6 Models Checkpoints ---
    # 建议确认这些路径是否真实存在
    default_ckpts = (
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' 
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_gfnet_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' 
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_ffc_1201/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' 
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_deformationfno_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' 
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_fsel_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth,' 
        '/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_p2_4x2_spect_1130/mamba_unrolled/mamba_unrolled_best_ssim_model.pth'
    )
    ap.add_argument("--ckpts", type=str, default=default_ckpts)
    ap.add_argument("--names", type=str, default="DAPSO,GFNet,FFC,DeformFNO,FSEL,SpectFormer")
    ap.add_argument("--model_types", type=str, default="dapso,gfnet,ffc,deformfno,fsel,spectformer")

    # --- Plotting Args ---
    ap.add_argument("--n_bins", type=int, default=256)
    ap.add_argument("--max_items", type=int, default=None, help="Debug usage: only run first N images")
    ap.add_argument("--logy", action="store_true", help="Use log scale for Y axis")
    ap.add_argument("--out", type=str, default="radial_error_spectrum_06.pdf")

    args = ap.parse_args()

    # 1. 设置画图风格
    set_publication_style()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 2. 解析参数
    ckpts = [x.strip() for x in args.ckpts.split(",") if x.strip()]
    names = [x.strip() for x in args.names.split(",") if x.strip()]
    mtypes = [x.strip() for x in args.model_types.split(",") if x.strip()]
    
    # 处理 None
    mtypes = [None if mt.lower() in ("none", "null", "") else mt for mt in mtypes]

    if len(ckpts) != 6: raise ValueError(f"Need 6 ckpts, got {len(ckpts)}")
    if len(names) != 6: raise ValueError(f"Need 6 names, got {len(names)}")

    # 3. 加载模型
    base = argparse.Namespace(model=args.model, cfg=args.cfg, patch_size=args.patch_size)
    base = ensure_args_for_config(base)
    
    models = []
    print(">>> Loading Models...")
    for i in range(6):
        print(f"   Loading [{names[i]}] ...")
        m = build_model(base, device, model_type=mtypes[i])
        load_weights(m, ckpts[i], device)
        models.append(m)

    # 4. 数据集
    print(f">>> Loading Dataset: {args.data_dir}")
    dataset = SliceData_CC359(
        data_dir=args.data_dir, acceleration=args.acceleration,
        mask_type=args.mask_type, resolution=args.resolution, type=args.split,
    )
    N = len(dataset)
    use_N = N if args.max_items is None else min(N, int(args.max_items))
    print(f"    Total: {N}, Using: {use_N}")

    # 5. 计算循环
    binner = None
    sum_prof = [None] * 6
    count = 0

    # 尝试使用 tqdm
    try:
        from tqdm import tqdm
        iterator = tqdm(range(use_N), desc="Inferencing")
    except ImportError:
        iterator = range(use_N)

    with torch.no_grad():
        for idx in iterator:
            sample = dataset[idx]
            
            # Prepare inputs
            us_image = to_4d(to_tensor(sample["us_image"])).to(device)
            us_mask  = to_4d(to_tensor(sample["us_mask"])).to(device)
            coil_map = to_4d(to_tensor(sample["coil_map"])).to(device)
            gt_c = to_complex2d(sample["fs_image"])

            # Init Binner
            if binner is None:
                H, W = gt_c.shape
                binner = RadialBinner(H, W, n_bins=args.n_bins)
                for i in range(6):
                    sum_prof[i] = np.zeros((args.n_bins,), dtype=np.float64)

            # Inference Loop
            for i, m in enumerate(models):
                out = unwrap_output(m(us_image, us_mask, coil_map))
                pred_c = to_complex2d(out)
                prof = error_spectrum_profile(pred_c, gt_c, binner)
                sum_prof[i] += prof
            
            count += 1

    # 6. 绘图 (核心修改部分)
    print(">>> Plotting...")
    fig, ax = plt.subplots(figsize=(8, 6)) # 8x6 英寸

    r = binner.r
    
    # 样式定义
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    # 高对比度颜色组
    colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b'] 
    # d62728是红色，作为主方法

    MAIN_IDX = 0 # 假设第一个模型(DAPSO)是你的方法

    # 预计算 Y 轴的最大值（仅在 0-0.6 范围内）
    # 这样可以防止 Y 轴被 0.6 以外的数据撑得太大，导致 0-0.6 部分看起来很扁
    y_max_in_window = -1.0
    valid_mask = r <= 0.6

    for i in range(6):
        mean_prof = sum_prof[i] / max(count, 1)
        
        # 记录可视范围内的最大值
        local_max = np.max(mean_prof[valid_mask])
        if local_max > y_max_in_window:
            y_max_in_window = local_max

        # 样式逻辑
        if i == MAIN_IDX:
            lw = 3.5      # 主方法非常粗
            alpha = 1.0
            z = 100
            color = colors[0] # 红色
        else:
            lw = 2.0      # 其他方法适中
            alpha = 0.85
            z = 10
            color = colors[i]

        ax.plot(
            r, mean_prof,
            linewidth=lw,
            alpha=alpha,
            color=color,
            linestyle=linestyles[i % len(linestyles)],
            label=names[i],
            zorder=z
        )

    # === 关键修改：只显示 0.0 到 0.6 ===
    ax.set_xlim(0.0, 0.6)
    
    # 如果是非对数坐标，手动调整一下 Y 轴上限，留点余量
    if not args.logy:
        ax.set_ylim(0.0, y_max_in_window * 1.1)
    else:
        ax.set_yscale("log")

    # 标签与网格
    ax.set_xlabel("Normalized Frequency Radius")
    ax.set_ylabel("Error Spectrum Energy (Log)" if args.logy else "Error Spectrum Energy")
    ax.grid(True, which="both", ls="-", alpha=0.3)

    # 图例
    ax.legend(
        loc="upper right",
        handlelength=3.0, # 图例线长一点，看清虚线
        borderpad=0.5,
        labelspacing=0.4
    )

    fig.tight_layout()

    # 保存文件
    out_path = args.out
    if not out_path.endswith(".pdf"):
        out_path = os.path.splitext(out_path)[0] + ".pdf"
    
    # bbox_inches='tight' 防止字号太大导致边缘被裁
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"[Done] Saved to: {out_path}")
    print(f"       X-Axis Range: 0.0 - 0.6")

if __name__ == "__main__":
    main()
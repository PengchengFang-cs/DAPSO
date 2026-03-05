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
# 针对“四宫格拼图”的绘图风格配置
# ----------------------------
def set_publication_style():
    """
    针对 2x2 拼图优化的风格：
    字号极大，线条极粗，确保缩小后依然清晰可见。
    """
    plt.rcParams.update({
        # 字体：优先使用 Times New Roman (论文标准)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        
        # 字号：设置得非常大
        'font.size': 20,              # 全局基础字号
        'axes.labelsize': 24,         # XY轴标签字号 (缩小后才看得清)
        'axes.titlesize': 24,         # 标题字号
        'xtick.labelsize': 20,        # 刻度字号
        'ytick.labelsize': 20,
        
        # 线条：加粗
        'axes.linewidth': 2.5,        # 坐标轴边框粗细
        'lines.linewidth': 3.5,       # 曲线粗细 (非常重要)
        'xtick.major.width': 2.5,     # 刻度线粗细
        'ytick.major.width': 2.5,
        'grid.linewidth': 1.5,
        
        # 图例
        'legend.fontsize': 18,
        'legend.frameon': False,      # 去掉图例边框，减少杂乱感
    })


# ----------------------------
# 基础工具函数 (保持不变)
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
    incompatible = model.load_state_dict(sd, strict=False)
    if incompatible.missing_keys:
        # 只打印简略信息，防止刷屏
        print(f"[Warn] Missing keys: {len(incompatible.missing_keys)}")
    print(f"[OK] Loaded: {os.path.basename(ckpt_path)}")

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

def list_dapso_modules(model):
    mods = []
    for name, m in model.named_modules():
        if hasattr(m, "UC_h") and hasattr(m, "_hc_branch"):
            mods.append((name, m))
    return mods

def estimate_M_per_freq(Zin, Zout, ridge=1e-4):
    Zin = Zin.to(torch.complex64)
    Zout = Zout.to(torch.complex64)
    N, C, L = Zin.shape
    M = torch.zeros((L, C, C), dtype=torch.complex64, device=Zin.device)
    I = torch.eye(C, dtype=torch.complex64, device=Zin.device)
    for k in range(L):
        X = Zin[:, :, k]
        Y = Zout[:, :, k]
        XtX = X.conj().T @ X
        XtY = X.conj().T @ Y
        A = torch.linalg.solve(XtX + ridge * I, XtY)
        M[k] = A
    return M

def offdiag_ratio(M_lcc):
    Mabs2 = (M_lcc.abs() ** 2)
    total = Mabs2.sum(dim=(1, 2))
    diag = (torch.diagonal(Mabs2, dim1=1, dim2=2)).sum(dim=1)
    off = torch.clamp(total - diag, min=0.0)
    return torch.sqrt(off / (total + 1e-12)).detach().cpu().numpy()

@torch.no_grad()
def compute_offdiag_curves_for_hc_branch(dapso, model, loader, device="cuda",
                                         max_batches=20, max_samples_per_batch=4096,
                                         ridge=1e-4):
    dapso.eval()
    model.eval()
    uc = dapso.UC_h
    sum_can = None
    sum_learn = None
    count = 0
    L_ref = None
    cache = {"x_in": None}
    def pre_hook(module, inputs):
        cache["x_in"] = inputs[0].detach()
    h = dapso.register_forward_pre_hook(pre_hook)

    nb = 0
    for batch in loader:
        nb += 1
        if nb > max_batches: break
        us_image = batch["us_image"].to(device)
        us_mask  = batch["us_mask"].to(device)
        coil_map = batch["coil_map"].to(device)
        cache["x_in"] = None
        _ = model(us_image, us_mask, coil_map)
        x = cache["x_in"]
        if x is None: continue
        x = x.to(device)
        B, C, H, W = x.shape
        xw = x.permute(0, 3, 1, 2).reshape(B * W, C, H)
        if max_samples_per_batch is not None and xw.shape[0] > max_samples_per_batch:
            ridx = torch.randperm(xw.shape[0], device=device)[:max_samples_per_batch]
            xw = xw[ridx]
        Zin_can = torch.fft.fft(xw, dim=-1)
        Zin_learn = uc(Zin_can, inverse=False)
        y = dapso._hc_branch(x)
        yw = y.permute(0, 3, 1, 2).reshape(B * W, C, H)
        if max_samples_per_batch is not None and yw.shape[0] > max_samples_per_batch:
            ridx = torch.randperm(yw.shape[0], device=device)[:max_samples_per_batch]
            yw = yw[ridx]
        Zout_can = torch.fft.fft(yw, dim=-1)
        Zout_learn = uc(Zout_can, inverse=False)
        M_can = estimate_M_per_freq(Zin_can, Zout_can, ridge=ridge)
        M_learn = estimate_M_per_freq(Zin_learn, Zout_learn, ridge=ridge)
        r_can = offdiag_ratio(M_can)
        r_learn = offdiag_ratio(M_learn)
        if sum_can is None:
            L_ref = r_can.shape[0]
            sum_can = np.zeros((L_ref,), dtype=np.float64)
            sum_learn = np.zeros((L_ref,), dtype=np.float64)
        sum_can += r_can
        sum_learn += r_learn
        count += 1
    h.remove()
    if count == 0: return None, None, None
    omega = np.linspace(-1.0, 1.0, L_ref)
    return omega, (sum_can / count), (sum_learn / count)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # Dataset & Model params
    ap.add_argument("--data_dir", type=str, default='/scratch/pf2m24/data/CCP359/Val')
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--acceleration", type=int, default=8)
    ap.add_argument("--mask_type", type=str, default="equispaced")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--model", type=str, default="mamba_unrolled")
    ap.add_argument("--cfg", type=str, default="../code/configs/vmamba_tiny.yaml")
    ap.add_argument("--patch_size", type=int, default=2)
    ap.add_argument("--model_type", type=str, default="dapso")
    ap.add_argument("--ckpt", type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_ssim_model.pth')
    
    # Layers to plot
    ap.add_argument("--layer_idxs", type=str, default="0,2,4,6")

    # Compute params
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_batches", type=int, default=50)
    ap.add_argument("--max_samples_per_batch", type=int, default=2048)
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--out_prefix", type=str, default="verify_theorem")

    args = ap.parse_args()
    ensure_args_for_config(args)

    # 1. 设置大图风格
    set_publication_style()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    print(">>> Building model & Loading weights...")
    model = build_model(args, device)
    load_weights(model, args.ckpt, device)

    dapso_list = list_dapso_modules(model)
    print(f">>> Found {len(dapso_list)} DAPSO modules.")

    target_idxs = [int(x) for x in args.layer_idxs.split(",") if x.strip()]
    
    dataset = SliceData_CC359(
        data_dir=args.data_dir,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        resolution=args.resolution,
        type=args.split,
    )

    for idx in target_idxs:
        if idx >= len(dapso_list): continue

        name, dapso = dapso_list[idx]
        print(f"\n=== Processing Layer {idx} ===")

        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        omega, r_can, r_learn = compute_offdiag_curves_for_hc_branch(
            dapso, model, loader, device=device,
            max_batches=args.max_batches,
            max_samples_per_batch=args.max_samples_per_batch,
            ridge=args.ridge,
        )

        if omega is None: continue

        # ----------------------------------------------------
        # 绘图逻辑优化：更紧凑的画布，更大的字体
        # ----------------------------------------------------
        # figsize=(5, 4) 是关键。画布越小，设定的 font.size=20 看起来就越大。
        # 如果你用 (10, 8)，20号字看起来就很小了。
        fig, ax = plt.subplots(figsize=(5, 4)) 
        
        # 绘制曲线 (Gray + Red 高对比度)
        ax.plot(omega, r_can, linewidth=3.0, linestyle='--', color='gray', alpha=0.7, label="Canonical")
        ax.plot(omega, r_learn, linewidth=3.5, color='#d62728', alpha=1.0, label="Learned")
        
        # 设置标题 (用 Layer X 简单标识即可)
        ax.set_title(f"Layer {idx}", pad=10, fontweight='bold')
        
        # 标签
        ax.set_xlabel(r"Normalized Frequency $\omega$")
        ax.set_ylabel("Off-diag Ratio") # 稍微简写一点，防止挤占空间
        
        # 坐标轴范围和网格
        ax.grid(True, linestyle='-', alpha=0.3)
        # 根据你的数据，通常 ratio 在 0~1 之间，如果有固定范围可以解开下面这行
        # ax.set_ylim(0, 1.05)
        
        # 图例：只在第一张图（比如Layer 0）显示图例，或者每张图显示极简图例
        # 为了四宫格整洁，建议把图例设得小一点，或者放在最不遮挡的地方
        ax.legend(loc="upper right", fontsize=16, handlelength=1.5)
        
        # 极度紧凑布局，去除白边
        fig.tight_layout()
        
        # 保存为 PDF (矢量图，强烈推荐)
        out_name = f"{args.out_prefix}_layer_{idx}.pdf"
        fig.savefig(out_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[Saved] {out_name}")

    print("\n[Done] For LaTeX, use \\includegraphics[width=0.48\\linewidth]{...}")

if __name__ == "__main__":
    main()
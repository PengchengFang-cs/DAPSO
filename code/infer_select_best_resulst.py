#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
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
# from dataloaders.CC359_dataset_PGIUN_4 import SliceData_CC359

# =========================
# Model builder (与训练一致)
# =========================
def build_model(args, device):
    if args.model == "mamba_unrolled":
        from networks.vision_mamba import MambaUnrolled as VIM_seg
    elif args.model == "mamba_unet":
        from networks.vision_mamba import MambaUnet as VIM_seg
    elif args.model == "swin_unet":
        from networks.vision_transformer import SwinUnet as VIM_seg
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    elif args.model == "swin_unrolled":
        from networks.vision_transformer import SwinUnrolled as VIM_seg
        if args.cfg is None:
            args.cfg = "../code/configs/swin_tiny_patch4_window7_224_lite.yaml"
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    config = get_config(args)
    model = VIM_seg(config, patch_size=args.patch_size, num_classes=2).to(device)
    model.eval()
    return model


# =========================
# Metrics (与你原来一致)
# =========================
def nmse(gt, pred):
    return np.linalg.norm(gt - pred) ** 2 / (np.linalg.norm(gt) ** 2 + 1e-12)

def psnr(gt, pred):
    # 与你原来一致：data_range = (max-min)
    return peak_signal_noise_ratio(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))

def ssim(gt, pred):
    return structural_similarity(gt, pred, data_range=max(gt.max() - gt.min(), 1e-12))


# =========================
# Checkpoint loading
# =========================
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

    # 仅允许 BN buffer 类缺失（如果你的 ckpt 完整，这里通常为空）
    if missing:
        only_bn = all(("running_mean" in k or "running_var" in k or "num_batches_tracked" in k) for k in missing)
        if not only_bn:
            raise RuntimeError(f"Missing keys (not only BN buffers): {missing}")
        else:
            print("[Warn] Missing BN buffer keys:", missing)

    if unexpected:
        print("[Warn] Unexpected keys:", unexpected)

    print(f"[OK] Loaded weights from: {ckpt_path}")


# =========================
# Visualization (vmax 不改)
# =========================
def save_compare_figure(us_mag, recon_mag, gt_mag, save_path):
    err = np.abs(recon_mag - gt_mag)
    vmax = max(us_mag.max(), recon_mag.max(), gt_mag.max())  # 你原来的 vmax 逻辑（不改）

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(us_mag, cmap='gray', vmin=0, vmax=vmax);    axes[0].set_title("Under (mag)");  axes[0].axis('off')
    axes[1].imshow(recon_mag, cmap='gray', vmin=0, vmax=vmax); axes[1].set_title("Recon (mag)");  axes[1].axis('off')
    axes[2].imshow(gt_mag, cmap='gray', vmin=0, vmax=vmax);    axes[2].set_title("Target (mag)"); axes[2].axis('off')
    axes[3].imshow(err, cmap='magma');                         axes[3].set_title("Error");        axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def tensor_to_mag(x):
    """
    接受 torch.Tensor 或 numpy.ndarray，返回 2D numpy 幅度图 [H,W]
    兼容形状：
      [B,2,H,W] -> sqrt(real^2+imag^2) 取第一个batch
      [B,1,H,W] -> 取第一个batch的单通道
      [2,H,W]   -> 实虚
      [1,H,W]   -> 单通道
      [H,W]     -> 已是2D
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if x.ndim == 4 and x.shape[0] == 1:
        x = x[0]

    if x.ndim == 3:
        C, H, W = x.shape
        if C == 2:
            mag = torch.sqrt(x[0]**2 + x[1]**2)
            return mag.cpu().numpy()
        elif C == 1:
            return x[0].cpu().numpy()
        else:
            raise ValueError(f"Unexpected channels: {C}, expect 1 or 2.")
    elif x.ndim == 2:
        return x.cpu().numpy()
    else:
        raise ValueError(f"Unexpected shape: {tuple(x.shape)}")

def to_4d(x: torch.Tensor, expect_c: int = None) -> torch.Tensor:
    # x: [H,W] or [C,H,W] or [B,C,H,W] -> [1,C,H,W]
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected ndim={x.ndim}, shape={tuple(x.shape)}")

    if expect_c is not None:
        assert x.shape[1] == expect_c, f"Channel mismatch: got {x.shape[1]}, expect {expect_c}"
    return x


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # —— 与训练保持一致的关键参数 ——
    parser.add_argument('--model', type=str, default='mamba_unrolled',
                        choices=['mamba_unrolled', 'mamba_unet', 'swin_unet', 'swin_unrolled'])
    parser.add_argument('--cfg', type=str, default='../code/configs/vmamba_tiny.yaml')
    parser.add_argument('--patch_size', type=int, default=2)

    # —— 推理相关 ——
    parser.add_argument('--ckpt', type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/model/mamba_unrolled_140_Patch_2_cc359_8x_daspo_1227/mamba_unrolled/mamba_unrolled_best_psnr_model.pth')
    parser.add_argument('--data_dir', type=str, default='/scratch/pf2m24/data/CCP359/Val',
                        help='folder with .npy volumes for CC359')
    parser.add_argument('--save_dir', type=str, default='/scratch/pf2m24/projects/MRIRecon/Dual_Axis/results/cc359_8x',
                        help='where to save outputs')

    # —— dataset 参数（跟你之前一致，可改）——
    parser.add_argument('--acceleration', type=int, default=8)
    parser.add_argument('--mask_type', type=str, default='equispaced')
    parser.add_argument('--resolution', type=int, default=256)
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])

    # —— TopK 选择 ——
    parser.add_argument('--topk', type=int, default=10)

    # —— 可选：是否额外存单张 under/recon/target/error ——（vmax 逻辑保持你原来的 max）
    parser.add_argument('--save_single', default=True)

    args = parser.parse_args()

    # 补齐 get_config/update_config 可能访问的字段，避免 AttributeError（照你原来写法）
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
        if not hasattr(args, k):
            setattr(args, k, v)

    os.makedirs(args.save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device:", device)

    # 1) Model + weights
    model = build_model(args, device)
    load_weights(model, args.ckpt, device)
    model.eval()

    # 2) Dataset
    dataset = SliceData_CC359(
        data_dir=args.data_dir,
        acceleration=args.acceleration,
        mask_type=args.mask_type,
        resolution=args.resolution,
        type=args.split,
    )
    N = len(dataset)
    print(f"Dataset size: {N}")

    # tqdm（没有也能跑）
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs): return x

    # 3) 遍历全数据集：用 PSNR 排序取 TopK（仅用 PSNR 做筛选，速度更快）
    K = max(1, int(args.topk))
    heap = []  # min-heap: (psnr, idx)

    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        for idx in tqdm(range(N), desc="Scanning (PSNR)"):
            sample = dataset[idx]

            us_image = to_4d(sample['us_image'], expect_c=2).to(device)  # [1,2,H,W]
            us_mask  = to_4d(sample['us_mask'],  expect_c=1).to(device)  # [1,1,H,W]
            coil_map = to_4d(sample['coil_map'], expect_c=1).to(device)  # [1,1,H,W]
            fs_image = sample['fs_image']  # numpy [2,H,W] (用于 gt)

            pred = model(us_image, us_mask, coil_map)      # [1,2,H,W]
            pred = pred[0].detach().cpu().numpy()          # [2,H,W]

            recon_mag = tensor_to_mag(pred)
            gt_mag    = tensor_to_mag(fs_image)

            p = psnr(gt_mag, recon_mag)

            if len(heap) < K:
                heapq.heappush(heap, (p, idx))
            else:
                if p > heap[0][0]:
                    heapq.heapreplace(heap, (p, idx))

    topk_list = sorted(heap, key=lambda x: x[0], reverse=True)  # [(psnr, idx), ...]
    print("\n===== TopK by PSNR =====")
    for r, (p, idx) in enumerate(topk_list, 1):
        print(f"Rank {r:02d}: idx={idx}  PSNR={p:.4f}")

    # 保存 topk 列表
    topk_root = os.path.join(args.save_dir, f"topk{K}")
    os.makedirs(topk_root, exist_ok=True)

    txt_path = os.path.join(topk_root, "topk_list.txt")
    with open(txt_path, "w") as f:
        for r, (p, idx) in enumerate(topk_list, 1):
            f.write(f"{r},{idx},{p:.6f}\n")
    print("Saved:", txt_path)

    # 4) 对 TopK 再跑一遍：存图 + 存 npy + 记录完整指标(PSNR/SSIM/NMSE)
    csv_path = os.path.join(topk_root, "topk_metrics.csv")
    with open(csv_path, "w") as fcsv:
        fcsv.write("rank,index,psnr,ssim,nmse,dir\n")

        with torch.no_grad():
            for r, (p_scan, idx) in enumerate(tqdm(topk_list, desc="Saving TopK"), 1):
                sample = dataset[idx]

                us_image = to_4d(sample['us_image'], expect_c=2).to(device)
                us_mask  = to_4d(sample['us_mask'],  expect_c=1).to(device)
                coil_map = to_4d(sample['coil_map'], expect_c=1).to(device)
                fs_image = sample['fs_image']

                pred = model(us_image, us_mask, coil_map)
                pred = pred[0].detach().cpu().numpy()

                us_mag    = tensor_to_mag(us_image)
                recon_mag = tensor_to_mag(pred)
                gt_mag    = tensor_to_mag(fs_image)

                p = psnr(gt_mag, recon_mag)
                s = ssim(gt_mag, recon_mag)
                n = nmse(gt_mag, recon_mag)

                out_dir = os.path.join(topk_root, f"rank{r:02d}_idx{idx:06d}_psnr{p:.2f}")
                os.makedirs(out_dir, exist_ok=True)

                # npy
                np.save(os.path.join(out_dir, "under.npy"), us_mag)
                np.save(os.path.join(out_dir, "recon.npy"), recon_mag)
                np.save(os.path.join(out_dir, "target.npy"), gt_mag)

                # 4联图（vmax 不改）
                save_compare_figure(
                    us_mag, recon_mag, gt_mag,
                    save_path=os.path.join(out_dir, "recon_compare.png")
                )

                # 可选：单图（vmax 保持你原来写法 max(arr.max(), 1.0)）
                if args.save_single:
                    plt.imsave(os.path.join(out_dir, "under.png"),  us_mag,    cmap='gray', vmin=0, vmax=max(us_mag.max(), 1.0))
                    plt.imsave(os.path.join(out_dir, "recon.png"),  recon_mag, cmap='gray', vmin=0, vmax=max(recon_mag.max(), 1.0))
                    plt.imsave(os.path.join(out_dir, "target.png"), gt_mag,    cmap='gray', vmin=0, vmax=max(gt_mag.max(), 1.0))
                    plt.imsave(os.path.join(out_dir, "error.png"),  np.abs(recon_mag - gt_mag), cmap='magma')

                fcsv.write(f"{r},{idx},{p:.6f},{s:.6f},{n:.8e},{out_dir}\n")

    print("Saved:", csv_path)
    print("All outputs in:", os.path.abspath(topk_root))


if __name__ == "__main__":
    main()

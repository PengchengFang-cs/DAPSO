import os
import argparse
import random
import logging
from typing import Optional, Callable, Dict, Any

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from datasets import load_dataset


# =========================
#   ADE20K Reconstruction Dataset (parquet)
# =========================
class ADE20KParquetRecon(Dataset):
    """
    ADE20K parquet 重建任务 Dataset.
    - input:  被降质后的图像（模型输入）
    - target: 干净原图（重建目标）

    hf_ds:    HuggingFace Dataset split（train / val）
    img_size: 输出分辨率 (img_size x img_size)
    degrade_fn: (B,C,H,W)->(B,C,H,W) 退化函数；None 时使用默认下采样+上采样。
    """

    def __init__(
        self,
        hf_ds,
        img_size: int = 256,
        degrade_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.ds = hf_ds
        self.img_size = img_size
        self.degrade_fn = degrade_fn

        self.to_tensor = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self) -> int:
        return len(self.ds)

    @torch.no_grad()
    def _default_degrade(self, x: torch.Tensor) -> torch.Tensor:
        """
        默认退化：简单 bicubic 下采样 + 上采样
        x: (B, C, H, W)
        """
        b, c, h, w = x.shape
        scale = 4
        low_h, low_w = h // scale, w // scale
        low = F.interpolate(x, size=(low_h, low_w),
                            mode="bicubic", align_corners=False)
        rec = F.interpolate(low, size=(h, w),
                            mode="bicubic", align_corners=False)
        return rec

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.ds[idx]
        img = sample["image"]   # 你 process_ade20k.py 生成的列名

        # 可能是 PIL，也可能是 numpy
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        # GT
        target = self.to_tensor(img).clamp(0.0, 1.0)   # (C,H,W)
        target_b = target.unsqueeze(0)                 # (1,C,H,W)

        # input（退化版）
        if self.degrade_fn is not None:
            inp_b = self.degrade_fn(target_b)
        else:
            inp_b = self._default_degrade(target_b)

        inp = inp_b.squeeze(0).clamp(0.0, 1.0)

        return {
            "input": inp,        # 模型输入
            "target": target,    # 重建目标
        }


# =========================
#   Metrics
# =========================
def calc_psnr(target: np.ndarray, pred: np.ndarray) -> float:
    return peak_signal_noise_ratio(target, pred, data_range=1.0)


def calc_ssim_batch(target: np.ndarray, pred: np.ndarray) -> float:
    """
    target/pred: (B, C, H, W), range [0,1]
    对每个通道、每张图计算 SSIM 再平均
    """
    if target.ndim == 3:
        target = target[None, ...]
        pred = pred[None, ...]
    b, c, h, w = target.shape
    ssim_list = []
    for i in range(b):
        for j in range(c):
            ssim_val = structural_similarity(
                target[i, j], pred[i, j],
                channel_axis=None,
                data_range=1.0
            )
            ssim_list.append(ssim_val)
    return float(np.mean(ssim_list))


# =========================
#   Model 接口（你自己实现）
# =========================
def build_model(name: str, img_size: int = 256, config=None) -> nn.Module:
    """
    这里留给你自己实现。

    要求：
      - 输入:  (B, 3, img_size, img_size)
      - 输出:  (B, 3, img_size, img_size)
      - 输出范围建议在 [0,1]（可以在模型里用 sigmoid / clamp）

    示例接口：
      if name == "conv":
          return YourConvAE(...)
      elif name == "vit":
          return YourViTAE(...)
      else:
          raise ValueError(...)
    """
    if name == 'mamba_unet':
        from networks.vision_mamba import MambaUnet
        
        model = MambaUnet(config, img_size=img_size, num_classes=3)
        return model
    raise NotImplementedError("请在 build_model 里根据 name 返回你自己的 nn.Module")


# =========================
#   Train / Val Loop
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, device, epoch, max_epoch, log_interval=50):
    model.train()
    pbar = tqdm(loader, ncols=120, desc=f"[Train] Epoch {epoch}/{max_epoch}")
    total_loss = 0.0
    for i, batch in enumerate(pbar):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        pred = model(x)
        loss = F.l1_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_interval == 0:
            pbar.set_postfix({"loss": f"{loss.item():.4e}"})

    avg_loss = total_loss / len(loader)
    return avg_loss


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0

    for batch in tqdm(loader, ncols=100, desc="[Val]"):
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        pred = model(x).clamp(0.0, 1.0)

        y_np = y.detach().cpu().numpy()
        p_np = pred.detach().cpu().numpy()

        total_psnr += calc_psnr(y_np, p_np)
        total_ssim += calc_ssim_batch(y_np, p_np)

    avg_psnr = total_psnr / len(loader)
    avg_ssim = total_ssim / len(loader)
    return avg_psnr, avg_ssim


# =========================
#   Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ade_root", type=str, default='/scratch/pf2m24/data/ADE20K/ADE20K/data',
        help="存放 train-*.parquet / validation-*.parquet 的目录"
    )
    parser.add_argument(
        "--model", type=str, default="mamba_unet",
        help="模型名字（传给 build_model，用来区分你自己的不同 baseline）"
    )
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out_dir", type=str, default="./runs_ade20k_recon")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.out_dir, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info(str(args))

    set_seed(args.seed)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 1) 读取 parquet 为 HuggingFace Dataset =====
    dataset_dict = load_dataset(
        "parquet",
        data_files={
            "train": os.path.join(args.ade_root, "train-*.parquet"),
            "val":   os.path.join(args.ade_root, "validation-*.parquet"),
        },
    )
    train_hfds = dataset_dict["train"]
    val_hfds   = dataset_dict["val"]
    logging.info(f"Train samples: {len(train_hfds)}, Val samples: {len(val_hfds)}")
    logging.info(f"Columns: {train_hfds.column_names}")

    # ===== 2) 包成 PyTorch Dataset & Dataloader =====
    train_set = ADE20KParquetRecon(train_hfds, img_size=args.img_size, degrade_fn=None)
    val_set   = ADE20KParquetRecon(val_hfds,   img_size=args.img_size, degrade_fn=None)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ===== 3) 模型 & 优化器 =====
    config = get_config(args)
    model = build_model(args.model, img_size=args.img_size, config=config).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_psnr = -1.0
    best_ssim = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, epoch, args.epochs
        )
        scheduler.step()

        val_psnr, val_ssim = validate(model, val_loader, device)

        logging.info(
            f"[Epoch {epoch}] "
            f"Train Loss: {train_loss:.4e}, "
            f"Val PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_ssim = val_ssim
            save_path = os.path.join(args.out_dir, f"best_{args.model}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_psnr": best_psnr,
                "best_ssim": best_ssim,
            }, save_path)
            logging.info(f"==> New best model saved to {save_path}")

    logging.info(
        f"Training finished. Best PSNR: {best_psnr:.4f}, SSIM: {best_ssim:.4f}"
    )


if __name__ == "__main__":
    main()

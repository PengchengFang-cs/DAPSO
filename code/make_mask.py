import numpy as np
import imageio.v2 as imageio

def npy_to_gray_png(
    npy_path,
    save_path,
    vmin=None,
    vmax=None
):
    img = np.load(npy_path)
    img = img.squeeze()
    assert img.ndim == 2, f"Expect 2D array, got {img.shape}"

    # 强烈建议：统一归一化（否则不同图亮度不可比）
    if vmin is None:
        vmin = img.min()
    if vmax is None:
        vmax = img.max()

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin + 1e-8)
    img = (img * 255).astype(np.uint8)

    imageio.imwrite(save_path, img)

# 用法
npy_to_gray_png(
    '/scratch/pf2m24/projects/MambaIR/simple_mambair/data_loading/mask_2_320_af4.npy',
    "fastmri_4x.png",
    vmin=0,      # MRI 常用：固定 vmin/vmax
    vmax=1
)

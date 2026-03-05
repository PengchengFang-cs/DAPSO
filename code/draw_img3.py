import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_tensor_endswith(state_dict, suffix: str) -> torch.Tensor:
    for k, v in state_dict.items():
        if k.endswith(suffix) and torch.is_tensor(v):
            return k, v
    raise KeyError(f"Cannot find parameter key ending with '{suffix}'")


def build_gate_from_table(tab_kc: torch.Tensor, tgt_len: int, eps: float = 1e-6) -> np.ndarray:
    """
    Replicate your discrete gate:
      (K,C) -> interpolate along K to tgt_len -> softplus -> mean-normalize per channel
    Returns numpy (C, tgt_len)
    """
    tab_kc = tab_kc.float()
    g = tab_kc.t().unsqueeze(0)  # (1,C,K)
    g = F.interpolate(g, size=tgt_len, mode="linear", align_corners=True).squeeze(0)  # (C,tgt_len)
    g = F.softplus(g)
    g = g / (g.mean(dim=1, keepdim=True) + eps)
    return g.detach().cpu().numpy()


def plot_heatmap(gate_cN: np.ndarray, out: str, title: str, log_scale: bool = True):
    """
    gate_cN: (C, N)
    """
    C, N = gate_cN.shape
    show = np.log(gate_cN + 1e-6) if log_scale else gate_cN

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    im = ax.imshow(
        show,
        aspect="auto",
        origin="lower",
        extent=[-1.0, 1.0, 0, C - 1]
    )
    ax.set_xlabel("Normalized frequency $\\omega\\in[-1,1]$")
    ax.set_ylabel("Channel index")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("log(gain)" if log_scale else "gain")
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    plt.close(fig)
    print(f"[Saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--H", type=int, required=True)
    ap.add_argument("--W", type=int, required=True)
    ap.add_argument("--out_prefix", type=str, default="gate")
    ap.add_argument("--no_log", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    k1, hc_tab = find_tensor_endswith(state, "gamma_hc_tab")
    k2, wc_tab = find_tensor_endswith(state, "gamma_wc_tab")
    print(f"[Found] {k1} {tuple(hc_tab.shape)}")
    print(f"[Found] {k2} {tuple(wc_tab.shape)}")

    gate_hc = build_gate_from_table(hc_tab, tgt_len=args.H)  # (C,H)
    gate_wc = build_gate_from_table(wc_tab, tgt_len=args.W)  # (C,W)

    plot_heatmap(
        gate_hc,
        out=f"{args.out_prefix}_HC.png",
        title=f"HC gate (interpolated to H={args.H})",
        log_scale=not args.no_log
    )
    plot_heatmap(
        gate_wc,
        out=f"{args.out_prefix}_WC.png",
        title=f"WC gate (interpolated to W={args.W})",
        log_scale=not args.no_log
    )


if __name__ == "__main__":
    main()

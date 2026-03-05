# dapsolab/models/layers/dapso.py
import torch, torch.nn as nn, torch.nn.functional as F
from math import pi
import math

def normalized_omega(n):  # 归一化频率坐标 [-1,1]
    idx = torch.linspace(0, n-1, n)
    omega = 2*pi*idx/n
    return (omega/pi) - 1.0  # [-1,1]

# dapsolab/models/layers/local_conv.py
import torch.nn as nn

class LocalDWConv(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, k, padding=k//2, groups=dim)
        self.pw = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
    def forward(self, x):
        return self.pw(self.act(self.dw(x)))

class MLP1D(nn.Module):
    def __init__(self, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, grid):  # grid: (N,)
        return self.net(grid.view(-1,1))  # (N,out_dim)


class ChannelBasis(nn.Module):
    """
    U_C: 通道谱基
      - mode='identity'：不变换
      - mode='learned_ortho'：学习正交（前向用 QR 的正交副本；复数安全）
    """
    def __init__(self, C: int, mode: str = 'identity'):
        super().__init__()
        self.mode = mode
        if mode == 'learned_ortho':
            W = torch.empty(C, C)
            nn.init.orthogonal_(W)
            self.W = nn.Parameter(W)

    def forward(self, x, inverse: bool = False):  # x: (B, C, L)，real/complex
        if self.mode == 'identity':
            return x
        Q, _ = torch.linalg.qr(self.W)                     # 正交副本（可微）
        W = Q.conj().t() if inverse else Q                 # 逆：共轭转置
        W = W.to(dtype=x.dtype, device=x.device)           # 对齐 dtype/device
        return torch.einsum('cd,bdl->bcl', W, x)           # (B,C,L)

class DAPSO(nn.Module):
    """
    DAPSO: Dual-Axis Product-Spectrum Operator (HC/WC) + Local complement
    输入/输出: (B, C, H, W)
    gating_mode: 'continuous' | 'discrete'
      - continuous: 你原来的 MLP 连续频率门控
      - discrete  : 频率侧表 + 线性插值，分辨率无关
    """
    def __init__(self, dim:int, rank:int=8, band:int=None, local:str="conv",
                 basis:str="identity", axis: str = "both",
                 gating_mode: str = "continuous", disc_bins: int | tuple = 96):
        super().__init__()
        self.C = dim
        self.R = rank
        self.band = band
        self.axis = axis
        self.gating_mode = gating_mode

        # 连续门控用的 MLP（HC / WC）
        self.a_h = MLP1D(rank); self.b_c1 = MLP1D(rank)
        self.a_w = MLP1D(rank); self.b_c2 = MLP1D(rank)

        # 离散门控的频率侧表（按 (K, C) 存；运行时对 H/W 做线性插值）
        if isinstance(disc_bins, (tuple, list)):
            kh, kw = int(disc_bins[0]), int(disc_bins[1])
        else:
            kh = kw = int(disc_bins)
        self.disc_bins_h = kh
        self.disc_bins_w = kw
        # 小正值初始化，乘 softplus 后稳定 > 0
        # self.gamma_hc_tab = nn.Parameter(torch.full((kh, dim), 0.01))
        # self.gamma_wc_tab = nn.Parameter(torch.full((kw, dim), 0.01))

        c1 = math.log(math.e - 1.0)  # ≈ 0.5413249 = softplus^{-1}(1.0)
        self.gamma_hc_tab = nn.Parameter(torch.full((kh, dim), c1))
        self.gamma_wc_tab = nn.Parameter(torch.full((kw, dim), c1))

        # 通道谱基
        self.Uc = ChannelBasis(dim, mode=basis)

        # Local 补偿
        if local == "conv":
            self.local = LocalDWConv(dim, k=3)
        elif local == "attn":
            self.local = ChannelAttentionBlock(dim)
        else:
            self.local = nn.Identity()

        # 融合
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)

    # ---- 工具: 频率表 -> 目标长度 的 1D 线性插值（shape: (L,C)) ----
    @staticmethod
    def _interp_freq_table(tab_lc: torch.Tensor, tgt_len: int) -> torch.Tensor:
        # tab_lc: (K, C)  ->  (1, C, K) 线性插值到 (1, C, tgt_len) -> (tgt_len, C)
        g = tab_lc.t().unsqueeze(0)  # (1, C, K)
        g = F.interpolate(g, size=tgt_len, mode='linear', align_corners=True)
        return g.squeeze(0).t()      # (tgt_len, C)

    def _parse_band_to_K(self, band, N: int):
        if band is None:
            return None
        if isinstance(band, float):
            alpha = max(0.0, min(0.5, float(band)))
            K = int(round(alpha * (N / 2.0)))
        else:
            K = int(band)
        K = max(0, min(K, N // 2))
        return K

    def _bandlimit_mask(self, N: int, C: int, device):
        K = self._parse_band_to_K(self.band, N)
        if K is None or K == 0:
            return None
        m = torch.zeros(N, dtype=torch.float32, device=device)
        m[:K] = 1.0; m[-K:] = 1.0
        return m[:, None] * torch.ones(C, dtype=torch.float32, device=device)[None, :]

    # ---------------- 门控：HC / WC ----------------
    def _gate_hc(self, H, C, device):
        if self.gating_mode == "continuous":
            omega_h = normalized_omega(H).to(device)                 # (H,)
            lam_c   = torch.linspace(-1, 1, C, device=device)
            A = self.a_h(omega_h)                                    # (H,R)
            B = self.b_c1(lam_c)                                     # (C,R)
            G = F.softplus(torch.einsum('hr,cr->hc', A, B))          # (H,C)
        else:  # 'discrete'
            G = self._interp_freq_table(self.gamma_hc_tab.to(device), H)  # (H,C)
            G = F.softplus(G)

            # 现在加一行“每通道均值归一”：让 mean=1
            G = G / (G.mean(dim=0, keepdim=True) + 1e-6)

        M = self._bandlimit_mask(H, C, device)
        if M is not None:
            G = G * M
        return G  # (H,C)

    def _gate_wc(self, W, C, device):
        if self.gating_mode == "continuous":
            omega_w = normalized_omega(W).to(device)                 # (W,)
            lam_c   = torch.linspace(-1, 1, C, device=device)
            A = self.a_w(omega_w)                                    # (W,R)
            B = self.b_c2(lam_c)                                     # (C,R)
            G = F.softplus(torch.einsum('wr,cr->wc', A, B))          # (W,C)
        else:  # 'discrete'
            G = self._interp_freq_table(self.gamma_wc_tab.to(device), W)  # (W,C)
            G = F.softplus(G)

            # 现在加一行“每通道均值归一”：让 mean=1
            G = G / (G.mean(dim=0, keepdim=True) + 1e-6)
        return G  # (W,C)

    # ---------------- 两个轴的分支（full FFT） ----------------
    def _hc_branch(self, x):
        B,C,H,W = x.shape
        device = x.device
        xw = x.permute(0,3,1,2).reshape(B*W, C, H)       # (BW,C,H)

        z  = torch.fft.fft(xw, dim=-1)                   # complex
        z  = self.Uc(z, inverse=False)
        G  = self._gate_hc(H, C, device).t()             # (C,H)
        z  = z * G[None, ...]
        z  = self.Uc(z, inverse=True)
        y  = torch.fft.ifft(z, dim=-1).real

        y  = y.reshape(B, W, C, H).permute(0,2,3,1)      # (B,C,H,W)
        return y

    def _wc_branch(self, x):
        B,C,H,W = x.shape
        device = x.device
        xh = x.permute(0,2,1,3).reshape(B*H, C, W)       # (BH,C,W)

        z  = torch.fft.fft(xh, dim=-1)
        z  = self.Uc(z, inverse=False)
        G  = self._gate_wc(W, C, device).t()             # (C,W)
        z  = z * G[None, ...]
        z  = self.Uc(z, inverse=True)
        y  = torch.fft.ifft(z, dim=-1).real

        y  = y.reshape(B, H, C, W).permute(0,2,1,3)      # (B,C,H,W)
        return y

    def forward(self, x):
        if self.axis == "hc":
            y = self._hc_branch(x)
        elif self.axis == "wc":
            y = self._wc_branch(x)
        elif self.axis == 'half':
            x_hc, x_wc = torch.chunk(x, 2, dim=1)
            y = torch.cat([self._hc_branch(x_hc), self._wc_branch(x_wc)], dim=1)
        elif self.axis == 'hc_wc':
            y = self._hc_branch(x); y = self._wc_branch(y)
        elif self.axis == 'wc_hc':
            y = self._wc_branch(x); y = self._hc_branch(y)
        else:
            y = self._hc_branch(x) + self._wc_branch(x)

        y = self.local(y)
        return x + self.fuse(y)




class ChannelAttentionBlock(nn.Module):
    """
    通道注意力模块（Channel Attention Block）
    采用 1x1 卷积 和 3x3 卷积，然后加权输入特征
    """
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.relu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1x1(out)
        out = self.relu(out)
        out = self.conv3x3(out)
        
        return self.sigmoid(out) * x  # Hadamard product（逐元素相乘）

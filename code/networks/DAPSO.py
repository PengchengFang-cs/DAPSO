# dapsolab/models/layers/dapso.py
import torch, torch.nn as nn, torch.nn.functional as F
from math import pi
from typing import Dict, Any
from dataclasses import dataclass
from timm.models.layers import DropPath, trunc_normal_
from math import pi

def normalized_omega_rfft(N: int, device):
    """
    rFFT 半谱坐标：k=0..N//2, ω=2πk/N ∈ [0, π]，线性映射到 [-1,1]
    返回 shape: (N//2 + 1,)
    """
    k = torch.arange(0, N // 2 + 1, device=device, dtype=torch.float32)
    omega = 2 * pi * k / N
    return (omega / pi) - 1.0

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
    
class DWInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=5, stride=1):
        super(DWInvertedBottleneck, self).__init__()
        hidden_dim = in_channels * expansion_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
    
        self.block = nn.Sequential(
            # 1. Pointwise conv (expand)
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),

            # 2. Depthwise conv
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.GroupNorm(1, hidden_dim),
            nn.GELU(),

            # 3. Pointwise conv (project)
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GELU(),
        )
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_residual:
            h = self.block(x)
            return x + h
        else:
            return self.block(x)
        
def normalized_omega(n):  # 归一化频率坐标 [-1,1]
    idx = torch.linspace(0, n-1, n)
    omega = 2*pi*idx/n
    return (omega/pi) - 1.0  # [-1,1]


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
    通道谱基 U_C：
    mode='identity'：不变换；'learned_ortho'：学习正交；'graph_poly'：图谱多项式(置换等变)
    下面给出可用的两种：identity/learned_ortho（graph_poly 可按需扩展）
    """
    def __init__(self, C, mode='identity'):
        super().__init__()
        self.mode = mode
        if mode == 'learned_ortho':
            w = torch.empty(C, C); nn.init.orthogonal_(w)
            self.W = nn.Parameter(w)  # forward 中可做一次 QR 保障正交

    def forward(self, x, inverse=False):  # x: (B, C, L), may be complex
        if self.mode == 'identity':
            return x
        if self.mode == 'learned_ortho':
            with torch.no_grad():
                W_eff, _ = torch.linalg.qr(self.W)   # (C,C)
            W = W_eff.t() if inverse else W_eff
            W = W.to(dtype=x.dtype, device=x.device)
            return torch.einsum('cd,bdl->bcl', W, x)
            # # 对齐 dtype / device，避免 complex vs float 冲突
            # W = W.to(dtype=x.dtype, device=x.device)
            # assert W.shape[0] == W.shape[1] == x.shape[1], \
            #     f"[ChannelBasis] channel mismatch: W={tuple(W.shape)}, x={tuple(x.shape)}"
            # return torch.einsum('cd,bdl->bcl', W, x)
        raise NotImplementedError


class SimpleConv(nn.Module):
    def __init__(self, hidden_dim:int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class DAPSO(nn.Module):
    """
    DAPSO: Dual-Axis Product-Spectrum Operator (HC/WC) + Local complement
    输入/输出: (B, C, H, W)
    """
    def __init__(self, dim:int, rank:int=8, band:int=None, local:str="dw", basis:str="identity"):
        super().__init__()
        self.C = dim
        self.R = rank
        self.band = band  # 频率投影的带宽 K（可选）
        # 连续门控：a(ω), b(λ) for HC 和 WC
        self.a_h = MLP1D(rank); self.b_c1 = MLP1D(rank)  # for HC
        self.a_w = MLP1D(rank); self.b_c2 = MLP1D(rank)  # for WC
        # 通道谱基
        # self.Uc = ChannelBasis(dim // 2, mode=basis)
        self.UC_h = ChannelBasis(dim // 2, mode=basis)
        self.UC_w = ChannelBasis(dim // 2, mode=basis)

        # Local 补偿
        self.local = SimpleConv(dim)

        # self.local = nn.Sequential(
        #     nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
        #     nn.GELU(),
        #     nn.Conv2d(dim, dim, 1, bias=False)
        # )

        # 融合
        self.ca = ChannelAttentionBlock(dim)

    def _parse_band(self, band, N, which: str = "h"):
        """
        将 band 解析为整数 K（每侧保留 K 个频点）。
        - band: int / float(0~0.5] / Tensor / tuple/list
        - N: 轴向长度（H 或 W）
        - which: 'h' 或 'w'，当 band 是 tuple/list 时选用哪个
        """
        if band is None:
            return 0
        # 取维度
        val = band
        if isinstance(band, (tuple, list)):
            val = band[0] if which == "h" else band[1]
        if torch.is_tensor(val):
            val = val.item()

        # 解析到整数
        if isinstance(val, float):
            # 视为 Nyquist 比例，限定到 [0, 0.5]
            alpha = max(0.0, min(0.5, float(val)))
            k = int(round(alpha * (N / 2)))
        else:
            k = int(val)

        # clamp 到合法范围：每侧最多 N//2
        k = max(0, min(k, N // 2))
        return k

    def _bandlimit_mask_h(self, H, C):
        """
        rFFT 半谱带限：仅保留 0..K（含 DC），返回 (Hh, C)，Hh = H//2 + 1
        """
        Hh = H // 2 + 1
        K = self._parse_band(self.band, H, which="h")
        if K <= 0: return None
        K = min(K, Hh - 1)
        m_h = torch.zeros(Hh, dtype=torch.float32); m_h[:K+1] = 1.0
        m_c = torch.ones(C,  dtype=torch.float32)
        return m_h[:, None] * m_c[None, :]  # (Hh, C)

    def _bandlimit_mask_w(self, W, C):
        Wh = W // 2 + 1
        K = self._parse_band(getattr(self, "band_w", self.band), W, which="w")
        if K <= 0: return None
        K = min(K, Wh - 1)
        m_w = torch.zeros(Wh, dtype=torch.float32); m_w[:K+1] = 1.0
        m_c = torch.ones(C,  dtype=torch.float32)
        return m_w[:, None] * m_c[None, :]  # (Wh, C)


    def _gate_hc(self, H, C, device):
        Hh = H // 2 + 1
        omega_h = normalized_omega_rfft(H, device)    # (Hh,)
        lam_c   = torch.linspace(-1, 1, C, device=device)  # (C,)

        A = self.a_h(omega_h)                         # (Hh, R)
        B = self.b_c1(lam_c)                          # (C,  R)
        G = torch.einsum('hr,cr->hc', A, B)           # (Hh, C)
        G = F.softplus(G)
        G = G / (self.R ** 0.5)                       # 建议保留，便于不同 R 的公平性
        M = self._bandlimit_mask_h(H, C)
        if M is not None:
            G = G * M.to(device)
        return G                                       # (Hh, C)

    def _gate_wc(self, W, C, device):
        Wh = W // 2 + 1
        omega_w = normalized_omega_rfft(W, device)    # (Wh,)
        lam_c   = torch.linspace(-1, 1, C, device=device)

        A = self.a_w(omega_w)                         # (Wh, R)
        B = self.b_c2(lam_c)                          # (C,  R)
        G = torch.einsum('wr,cr->wc', A, B)           # (Wh, C)
        G = F.softplus(G)
        G = G / (self.R ** 0.5)
        M = self._bandlimit_mask_w(W, C)
        if M is not None:
            G = G * M.to(device)
        return G                                       # (Wh, C)


    def _hc_branch(self, x):
        B, C, H, W = x.shape
        device = x.device
        xw = x.permute(0, 3, 1, 2).reshape(B * W, C, H)     # (BW, C, H)

        z = torch.fft.rfft(xw, dim=-1, norm="ortho")        # (BW, C, Hh)
        z = self.UC_h(z, inverse=False)

        G = self._gate_hc(H, C, device).T                   # (C, Hh)
        z = z * G[None, ...]                                 # 广播乘（实门控自动提升为复数）

        z = self.UC_h(z, inverse=True)
        y = torch.fft.irfft(z, n=H, dim=-1, norm="ortho")   # ★ 指定 n=H，保证还原到原长度
        y = y.reshape(B, W, C, H).permute(0, 2, 3, 1)       # (B, C, H, W)
        return y

    def _wc_branch(self, x):
        B, C, H, W = x.shape
        device = x.device
        xh = x.permute(0, 2, 1, 3).reshape(B * H, C, W)     # (BH, C, W)

        z = torch.fft.rfft(xh, dim=-1, norm="ortho")        # (BH, C, Wh)
        z = self.UC_w(z, inverse=False)

        G = self._gate_wc(W, C, device).T                   # (C, Wh)
        z = z * G[None, ...]

        z = self.UC_w(z, inverse=True)
        y = torch.fft.irfft(z, n=W, dim=-1, norm="ortho")   # ★ 指定 n=W
        y = y.reshape(B, H, C, W).permute(0, 2, 1, 3)       # (B, C, H, W)
        return y


    def forward(self, x):
        hc_branch, wc_branch = torch.chunk(x, 2, dim=1)
        y = torch.cat([self._hc_branch(hc_branch), self._wc_branch(wc_branch)], dim=1) 
        y = self.ca(y)
        y = self.local(y)

        return x + y  # 残差融合

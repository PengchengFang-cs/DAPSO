# dapsolab/models/layers/dapso.py
import torch, torch.nn as nn, torch.nn.functional as F
from math import pi
import torch.nn as nn

def normalized_omega_rfft(N: int, device):
    """
    rFFT 半谱坐标：k = 0..N//2,  ω = 2πk/N ∈ [0, π]，线性映射到 [-1, 1]
    返回形状: (N//2 + 1,)
    """
    k = torch.arange(0, N // 2 + 1, device=device, dtype=torch.float32)
    omega = 2 * pi * k / N
    return (omega / pi) - 1.0


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
    def forward(self, x, inverse=False):  # x: (B, C, L)，可能是 complex
        if self.mode == 'identity':
            return x
        if self.mode == 'learned_ortho':
            with torch.no_grad():
                q, _ = torch.linalg.qr(self.W.data)  # (C,C)
                self.W.data.copy_(q)
            W = self.W.t() if inverse else self.W
            # ★ 对齐 dtype / device（关键）
            W = W.to(dtype=x.dtype, device=x.device)
            return torch.einsum('cd,bdl->bcl', W, x)
        raise NotImplementedError

class DAPSO(nn.Module):
    """
    DAPSO: Dual-Axis Product-Spectrum Operator (HC/WC) + Local complement
    输入/输出: (B, C, H, W)
    """
    def __init__(self, dim:int, rank:int=8, band:int=None, local:str="dw",
             basis:str="identity", axis: str = "both",
             gating_mode: str = "continuous",   # "continuous" | "discrete"
             disc_bins: int = None,             # 固定频率表长度（推荐 64/96）
             train_hw: tuple = None):           # 或者用训练分辨率推半谱长度 (H_train, W_train)
        super().__init__()
        self.C = dim
        self.R = rank
        self.band = band
        self.axis = axis
        self.gating_mode = gating_mode
        self.disc_bins = disc_bins
        self.train_hw = train_hw
        # 连续门控（原有）
        self.a_h = MLP1D(rank); self.b_c1 = MLP1D(rank)
        self.a_w = MLP1D(rank); self.b_c2 = MLP1D(rank)
        # ——离散门控的频率表（懒加载）——
        self.Ah_table = None   # (Lh0, R)
        self.Aw_table = None   # (Lw0, R)

        self.Uc = ChannelBasis(dim, mode=basis)
        self.local = LocalDWConv(dim, k=3) if local=="dw" else nn.Identity()
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)

    def _lazy_init_tables(self, H=None, W=None, device=None, dtype=torch.float32):
        """按 disc_bins / train_hw / 当前尺寸，懒加载频率表参数。"""
        if self.gating_mode != "discrete":
            return
        if self.Ah_table is None or self.Aw_table is None:
            if self.disc_bins is not None:
                Lh0 = int(self.disc_bins)
                Lw0 = int(self.disc_bins)
            elif self.train_hw is not None:
                H0, W0 = self.train_hw
                Lh0 = H0 // 2 + 1
                Lw0 = W0 // 2 + 1
            else:
                # 若都未提供，退化用当前尺寸
                assert H is not None and W is not None, \
                    "discrete gating needs disc_bins or train_hw or current H/W"
                Lh0 = H // 2 + 1
                Lw0 = W // 2 + 1
            Ah = torch.empty(Lh0, self.R, device=device, dtype=dtype)
            # Aw = torch.empty(Lw0, self.R, device=dtype, dtype=dtype)  # ← 注意这里的 device/dtype
            # 上一行有笔误，修正如下：
            Aw = torch.empty(Lw0, self.R, device=device, dtype=dtype)
            nn.init.xavier_uniform_(Ah)
            nn.init.xavier_uniform_(Aw)
            self.Ah_table = nn.Parameter(Ah)  # (Lh0, R)
            self.Aw_table = nn.Parameter(Aw)  # (Lw0, R)

    def _sample_table1d(self, table: torch.Tensor, L_target: int, mode: str = "linear") -> torch.Tensor:
        """
        频率表 1D 插值：table (L0, R) -> out (L_target, R)
        用 F.interpolate 在 L 轴上重采样：先转成 (1, R, L0)，再插值到 L_target。
        """
        L0, R = table.shape
        t = table.T.unsqueeze(0)                         # (1, R, L0)
        out = F.interpolate(t, size=L_target, mode=mode, align_corners=True)
        return out.squeeze(0).T                          # (L_target, R)

    def _parse_band_half(self, band, N_half):
        """
        将 band 解析为半谱上的 K（保留 0..K，含 DC）。
        - band: None / int / float(0~0.5]
        - N_half: 半谱长度 (N//2+1)
        """
        if band is None:
            return None
        if torch.is_tensor(band):
            band = band.item()
        if isinstance(band, float):
            # 比例按 Nyquist（N/2）计算；半谱上最大索引是 N//2
            K = int(round(band * ( (N_half - 1) )))  # N_half-1 对应 Nyquist
        else:
            K = int(band)
        # 安全裁剪：0..N_half-1
        K = max(0, min(K, N_half - 1))
        return K

    def _bandlimit_mask_h(self, H, C):
        """
        rFFT 半谱带限：返回 (Hh, C)，Hh = H//2 + 1；保留 0..K（含 DC）
        """
        Hh = H // 2 + 1
        K = self._parse_band_half(self.band, Hh)
        if K is None:
            return None
        m_h = torch.zeros(Hh, dtype=torch.float32)
        m_h[:K+1] = 1.0
        m_c = torch.ones(C, dtype=torch.float32)
        return m_h[:, None] * m_c[None, :]

    def _bandlimit_mask_w(self, W, C):
        Ww = W // 2 + 1
        # 你也可以支持 self.band_w；这里复用 self.band 做对称带限
        K = self._parse_band_half(self.band, Ww)
        if K is None:
            return None
        m_w = torch.zeros(Ww, dtype=torch.float32)
        m_w[:K+1] = 1.0
        m_c = torch.ones(C, dtype=torch.float32)
        return m_w[:, None] * m_c[None, :]

    def _gate_hc(self, H, C, device):
        Hh = H // 2 + 1
        lam_c = torch.linspace(-1, 1, C, device=device)   # (C,)
        B = self.b_c1(lam_c)                              # (C, R)

        if self.gating_mode == "discrete":
            self._lazy_init_tables(H=H, W=None, device=device, dtype=torch.float32)
            A = self._sample_table1d(self.Ah_table, Hh, mode="linear")   # (Hh, R)
        else:
            omega_h = normalized_omega_rfft(H, device)    # (Hh,)
            A = self.a_h(omega_h)                         # (Hh, R)

        G = torch.einsum('hr,cr->hc', A, B)               # (Hh, C)
        G = F.softplus(G)
        # 可选：不同 R 的轻度归一，便于 R 扫描公平
        # G = G / (self.R ** 0.5)

        M = self._bandlimit_mask_h(H, C)
        if M is not None:
            G = G * M.to(device)
        return G                                          # (Hh, C)

    def _gate_wc(self, W, C, device):
        Ww = W // 2 + 1
        lam_c = torch.linspace(-1, 1, C, device=device)
        B = self.b_c2(lam_c)                              # (C, R)

        if self.gating_mode == "discrete":
            self._lazy_init_tables(H=None, W=W, device=device, dtype=torch.float32)
            A = self._sample_table1d(self.Aw_table, Ww, mode="linear")   # (Ww, R)
        else:
            omega_w = normalized_omega_rfft(W, device)
            A = self.a_w(omega_w)                         # (Ww, R)

        G = torch.einsum('wr,cr->wc', A, B)               # (Ww, C)
        G = F.softplus(G)
        # G = G / (self.R ** 0.5)

        M = self._bandlimit_mask_w(W, C)
        if M is not None:
            G = G * M.to(device)
        return G

    def _hc_branch(self, x):
        # x: (B,C,H,W) -> 视作 W 组 (B*W,C,H)
        B, C, H, W = x.shape
        device = x.device
        xw = x.permute(0, 3, 1, 2).reshape(B * W, C, H)      # (BW,C,H)

        z = torch.fft.rfft(xw, dim=-1, norm="ortho")         # (BW,C,Hh) complex
        z = self.Uc(z, inverse=False)                        # (BW,C,Hh)

        G = self._gate_hc(H, C, device).T                    # (C,Hh)
        z = z * G[None, ...]                                 # 广播乘（实门控 -> 复数）

        z = self.Uc(z, inverse=True)
        y = torch.fft.irfft(z, n=H, dim=-1, norm="ortho")    # 指定 n=H，严格还原
        y = y.reshape(B, W, C, H).permute(0, 2, 3, 1)        # (B,C,H,W)
        return y

    def _wc_branch(self, x):
        # x: (B,C,H,W) -> 视作 H 组 (B*H,C,W)
        B, C, H, W = x.shape
        device = x.device
        xh = x.permute(0, 2, 1, 3).reshape(B * H, C, W)      # (BH,C,W)

        z = torch.fft.rfft(xh, dim=-1, norm="ortho")         # (BH,C,Ww) complex
        z = self.Uc(z, inverse=False)                        # (BH,C,Ww)

        G = self._gate_wc(W, C, device).T                    # (C,Ww)
        z = z * G[None, ...]

        z = self.Uc(z, inverse=True)
        y = torch.fft.irfft(z, n=W, dim=-1, norm="ortho")    # 指定 n=W
        y = y.reshape(B, H, C, W).permute(0, 2, 1, 3)        # (B,C,H,W)
        return y

    def forward(self, x):
        if self.axis == "hc":
            y = self._hc_branch(x)
        elif self.axis == "wc":
            y = self._wc_branch(x)
        elif self.axis == 'half':
            x_hc, x_wc = torch.chunk(x, 2, dim=1)              # C/2 + C/2
            y = torch.cat([self._hc_branch(x_hc),
                           self._wc_branch(x_wc)], dim=1)       # 恢复到 C
        elif self.axis == 'hc_wc':
            y = self._hc_branch(x)
            y = self._wc_branch(y)
        elif self.axis == 'wc_hc':
            y = self._wc_branch(x)
            y = self._hc_branch(y)
        else:
            y = self._hc_branch(x) + self._wc_branch(x)

        y = self.local(y)
        return x + self.fuse(y)  # 残差融合

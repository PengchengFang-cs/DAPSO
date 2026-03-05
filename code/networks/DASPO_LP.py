# dapsolab/models/layers/dapso.py
import torch, torch.nn as nn, torch.nn.functional as F
from math import pi

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
    def forward(self, x, inverse=False):  # x: (B, C, L)
        if self.mode == 'identity': return x
        if self.mode == 'learned_ortho':
            # 保障数值正交（代价小，精度够用）
            with torch.no_grad():
                q, _ = torch.linalg.qr(self.W.data)  # (C,C)
                self.W.data.copy_(q)
            W = self.W
            if inverse: W = W.t()
            return torch.einsum('cd,bdl->bcl', W, x)
        raise NotImplementedError

class DAPSO(nn.Module):
    """
    DAPSO: Dual-Axis Product-Spectrum Operator (HC/WC) + Local complement
    输入/输出: (B, C, H, W)
    """
    def __init__(self, dim:int, rank:int=8, band:int=None, basis:str="identity", axis: str = "both"):
        super().__init__()
        self.C = dim
        self.R = rank
        self.band = band  # 频率投影的带宽 K（可选）
        self.axis = axis  # 作用轴：'hc'/'wc'/'both'
        # 连续门控：a(ω), b(λ) for HC 和 WC
        self.a_h = MLP1D(rank); self.b_c1 = MLP1D(rank)  # for HC
        self.a_w = MLP1D(rank); self.b_c2 = MLP1D(rank)  # for WC
        # 通道谱基
        self.Uc = ChannelBasis(dim, mode=basis)

        # 融合
        self.fuse = nn.Conv2d(dim, dim, kernel_size=1)
        # 2个小控制标量
        self.gain_h, self.gain_w = nn.Parameter(torch.tensor(1.0)), nn.Parameter(torch.tensor(1.0))

    def _parse_band_to_K(self, band, N: int):
        """
        将 band 解析为整数 K（每侧保留 K 个频点）。
        - band: None | int | float(0~0.5]
        * int  : 直接视为 K
        * float: 视为 Nyquist 比例，K ≈ alpha * (N/2)
        - N: 轴向长度（H 或 W）
        """
        if band is None:
            return None
        # 标量解析
        if isinstance(band, float):
            alpha = max(0.0, min(0.5, float(band)))      # 裁剪到 [0, 0.5]
            K = int(round(alpha * (N / 2.0)))
        else:
            K = int(band)
        # 安全裁剪（每侧最多 N//2）
        K = max(0, min(K, N // 2))
        return K

    def _bandlimit_mask(self, N: int, C: int):
        """
        全谱 FFT 掩膜（对称两端）：返回 (N, C)
        支持 band 为 None / int / float(比例)
        """
        K = self._parse_band_to_K(self.band, N)
        if K is None or K == 0:
            return None
        m = torch.zeros(N, dtype=torch.float32)
        m[:K] = 1.0
        m[-K:] = 1.0
        mc = torch.ones(C, dtype=torch.float32)
        return m[:, None] * mc[None, :]

    def _gate_hc(self, H, C, device):
        # 连续频率坐标
        omega_h = normalized_omega(H).to(device)       # (H,)
        lam_c   = torch.linspace(-1, 1, C, device=device)  # 简化为索引归一化
        A = self.a_h(omega_h)          # (H,R)
        B = self.b_c1(lam_c)           # (C,R)
        G = torch.einsum('hr,cr->hc', A, B)  # (H,C)
        G = F.softplus(G)              # 仅做幅门控，稳定
        
        if self.band is not None:
            M = self._bandlimit_mask(H, C)                   # (H,C)
            if M is not None: G = G * M.to(device)
        return G  # (H,C)

    def _gate_wc(self, W, C, device):
        omega_w = normalized_omega(W).to(device)       # (W,)
        lam_c   = torch.linspace(-1, 1, C, device=device)
        A = self.a_w(omega_w)          # (W,R)
        B = self.b_c2(lam_c)           # (C,R)
        G = torch.einsum('wr,cr->wc', A, B)  # (W,C)
        G = F.softplus(G)
        if self.band is not None:
            M = self._bandlimit_mask(W, C)                   # (H,C)
            if M is not None: G = G * M.to(device)
        return G  # (W,C)

    def _hc_branch(self, x):
        # x: (B,C,H,W) -> 视作 W 组 (B*W,C,H)
        B,C,H,W = x.shape
        device = x.device
        xw = x.permute(0,3,1,2).reshape(B*W, C, H)      # (BW,C,H)
        z = torch.fft.fft(xw, dim=-1)                   # H 轴 FFT（复数）
        z = self.Uc(z, inverse=False)                   # 通道谱基 (BW,C,H)
        G = self._gate_hc(H, C, device).T               # (C,H)
        z = z * G[None, ...]                            # 广播乘
        
        z = self.Uc(z, inverse=True)
        z = z * self.gain_h # 补一个可学习标量
        y = torch.fft.ifft(z, dim=-1).real              # 回到实数
        y = y.reshape(B, W, C, H).permute(0,2,3,1)      # (B,C,H,W)
        return y

    def _wc_branch(self, x):
        # x: (B,C,H,W) -> 视作 H 组 (B*H,C,W)
        B,C,H,W = x.shape
        device = x.device
        xh = x.permute(0,2,1,3).reshape(B*H, C, W)      # (BH,C,W)
        z = torch.fft.fft(xh, dim=-1)                   # W 轴 FFT
        z = self.Uc(z, inverse=False)
        G = self._gate_wc(W, C, device).T               # (C,W)
        z = z * G[None, ...]
        
        z = self.Uc(z, inverse=True)
        y = torch.fft.ifft(z, dim=-1).real
        y = y.reshape(B, H, C, W).permute(0,2,1,3)      # (B,C,H,W)
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

        return x + self.fuse(y)  # 残差融合


# =======================
# 低通/下采样工具（可复用）
# =======================
import math
import torch, torch.nn as nn, torch.nn.functional as F

def binomial_1d_5():
    # [1, 4, 6, 4, 1] / 16
    k = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
    k = k / k.sum()
    return k  # (5,)

def gaussian_1d(kernel_size: int, sigma: float):
    assert kernel_size % 2 == 1 and kernel_size >= 3
    r = kernel_size // 2
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k  # (K,)

def _sep_conv(x: torch.Tensor, k1d: torch.Tensor, horizontal: bool, stride_hw=(1, 1)):
    """
    通道可分离的一维卷积（先 pad，再调用），groups=C。
    horizontal=True  => 1xK 卷积； False => Kx1 卷积
    """
    N, C, H, W = x.shape
    k1d = k1d.to(device=x.device, dtype=x.dtype)
    if horizontal:
        w = k1d.view(1, 1, 1, -1).repeat(C, 1, 1, 1)  # (C,1,1,K)
    else:
        w = k1d.view(1, 1, -1, 1).repeat(C, 1, 1, 1)  # (C,1,K,1)
    y = F.conv2d(x, w, bias=None, stride=stride_hw, padding=0, groups=C)
    return y

class SeparableReduce(nn.Module):
    """
    低通 + 下采样（2x）：pad -> 1xK -> Kx1(stride=2,2)
    注：这里的 stride=(2,2) 会一次性把 H/W 都下采样到 1/2。
    """
    def __init__(self, kernel_size=5, use_binomial=True, sigma=None, padding_mode="reflect"):
        super().__init__()
        self.pad = kernel_size // 2
        self.padding_mode = padding_mode
        if use_binomial:
            assert kernel_size == 5, "binomial_1d_5 仅支持 K=5"
            k1d = binomial_1d_5()
        else:
            assert sigma is not None, "Gaussian 需要 sigma"
            k1d = gaussian_1d(kernel_size, float(sigma))
        self.register_buffer("k1d", k1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,)*4, mode=self.padding_mode)
        x = _sep_conv(x, self.k1d, horizontal=True,  stride_hw=(1, 1))
        x = _sep_conv(x, self.k1d, horizontal=False, stride_hw=(2, 2))
        return x  # (B,C,H/2,W/2)

# =======================
# 方案 A：Laplacian-Pyramid DAPSO
# =======================
class DAPSO_LP(nn.Module):
    """
    Laplacian-Pyramid DAPSO:
      - 在低分辨率 low 子带上做全局频域（DAPSO）；
      - 高频 high 子带用轻量局部块补偿；
      - 融合后残差回传。
    依赖：上面的 DAPSO 类、SeparableReduce、LocalDWConv（已在文件内）
    """
    def __init__(self,
                 dim: int,
                 dapso_kwargs: dict,
                 up_mode: str = 'bilinear',
                 hf_block: str = 'dw',          # 'dw' | 'ibn'
                 ibn_expansion: int = 5,
                 use_binomial: bool = True,
                 ksize: int = 5,
                 padding_mode: str = "reflect"):
        """
        参数：
          - dim: 通道数
          - dapso_kwargs: 传给低分辨率 DAPSO 的参数字典
              * 建议：{'rank': R, 'band': None, 'local': 'none', 'basis': 'identity', 'axis': 'hc_wc'}
          - up_mode: 上采样插值方式（'bilinear'）
          - hf_block: 高频补偿块类型（'dw' 使用 LocalDWConv；'ibn' 使用倒残差）
        """
        super().__init__()
        self.dim = dim
        self.up_mode = up_mode

        # 低通 + 下采样
        self.reduce = SeparableReduce(kernel_size=ksize, use_binomial=use_binomial, padding_mode=padding_mode)

        # 低分辨率 DAPSO（建议 local='none'，避免重复局部补偿）
        # 若传入的 dapso_kwargs 未设置 local，默认覆盖为 'none'
        _kwargs = dict(dapso_kwargs)
        self.dapso_low = DAPSO(dim=dim, **_kwargs)

        # 高频局部补偿块
        if hf_block == 'ibn':
            self.hf_local = _DWInvertedBottleneck(in_channels=dim, out_channels=dim, expansion_ratio=ibn_expansion)
        else:
            self.hf_local = LocalDWConv(dim, k=3)

        # 可学习增益（低频增量 / 高频）
        self.gain_low  = nn.Parameter(torch.tensor(1.0))
        self.gain_high = nn.Parameter(torch.tensor(1.0))

        # 融合
        self.fuse = ChannelAttentionBlock(dim)

    @staticmethod
    def _low_up_high(x, reducer: nn.Module, up_mode='bilinear'):
        """
        Laplacian 分解（单层）：
          low_down: 低通+下采样
          low_up_ref: 上采样回原分辨率的参考低频
          high: 高频残差（x - low_up_ref）
        """
        low_down = reducer(x)  # (B,C,H/2,W/2)
        low_up_ref = F.interpolate(low_down, scale_factor=2, mode=up_mode, align_corners=False)
        high = x - low_up_ref
        return low_down, low_up_ref, high

    def forward(self, x):
        """
        输出：x + fuse( gain_low * Δlow_up + gain_high * HF_out )
        其中 Δlow_up = up( dapso_low(low_down) - low_down )
        """
        # 1) Laplacian 分解
        low_down, low_up_ref, high = self._low_up_high(x, self.reduce, self.up_mode)

        # 2) 低分辨率做 DAPSO（返回是 residual 结构：low_down + delta）
        low_out = self.dapso_low(low_down)
        delta_low = low_out - low_down               # 只取“增量”
        delta_low_up = F.interpolate(delta_low, scale_factor=2, mode=self.up_mode, align_corners=False)
        # delta_low_up = F.interpolate(low_out, scale_factor=2, mode=self.up_mode, align_corners=False)

        # 3) 高频仅作局部卷积
        high_out = self.hf_local(high)

        # 4) 融合 + 残差
        y = delta_low_up + high_out
        y = self.fuse(y)
        return x + y


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
    
# 备选：倒残差 IBN（轻量实现；仅在 hf_block='ibn' 时用）
class _DWInvertedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio=2):
        super().__init__()
        hidden = in_channels * expansion_ratio
        self.use_res = (in_channels == out_channels)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.GroupNorm(1, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, out_channels, 1, bias=False),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y if self.use_res else y
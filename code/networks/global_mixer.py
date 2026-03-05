import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------
# 1. GFNet-style global filter
# -------------------------------

class GFNetMixer2D(nn.Module):
    """
    GFNet-style global filter mixer (NCHW).
    Ref: GFNet - Global Filter Networks for Visual Recognition.

    - 理论上：FFT2 -> per-channel complex filter -> IFFT2。
    - 这里使用 rFFT/irFFT，并在一个粗网格 (Kh, Kw) 上学习频域 filter，
      运行时插值到当前频率分辨率 (H, W//2+1)，保持对任意 H,W 的适配能力。
    """

    def __init__(
        self,
        dim: int,
        bins=(64, 64),
        fft_norm: str = "ortho",
        residual: bool = True,
        use_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kh, self.kw = int(bins[0]), int(bins[1])
        self.fft_norm = fft_norm
        self.residual = residual

        # real & imag tables: (C, Kh, Kw)
        self.real = nn.Parameter(torch.zeros(dim, self.kh, self.kw))
        self.imag = nn.Parameter(torch.zeros(dim, self.kh, self.kw))

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dim, 1, 1))
        else:
            self.bias = None

        # 1x1 fuse after iFFT
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

        # small init as in GFNet
        nn.init.normal_(self.real, std=1e-3)
        nn.init.normal_(self.imag, std=1e-3)

    def _interp_filter(self, H: int, Wf: int, device, dtype):
        """
        Interpolate the learnable filter from (Kh, Kw) -> (H, Wf)
        in the frequency grid of rFFT: (H, W//2+1).
        """
        # (C,Kh,Kw) -> (1,C,Kh,Kw) for interpolate (NCHW)
        r = F.interpolate(
            self.real.unsqueeze(0),
            size=(H, Wf),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        i = F.interpolate(
            self.imag.unsqueeze(0),
            size=(H, Wf),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)
        # 用实数参数存 real/imag，组合成复数 filter
        filt = torch.complex(
            r.to(device=device, dtype=torch.float32),
            i.to(device=device, dtype=torch.float32),
        )  # (C,H,Wf)
        return filt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W) real
        """
        B, C, H, W = x.shape
        assert C == self.dim, f"Channel mismatch: {C} vs {self.dim}"

        # rFFT2 over spatial dims
        X = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)  # (B,C,H,Wf)
        _, _, _, Wf = X.shape

        Filt = self._interp_filter(H, Wf, x.device, X.dtype)  # (C,H,Wf)
        Y = X * Filt.unsqueeze(0)  # broadcast over batch

        # back to spatial
        y = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1), norm=self.fft_norm)

        if self.bias is not None:
            y = y + self.bias
        y = self.fuse(y)

        return x + y if self.residual else y


# -------------------------------
# 2. AFNO2D (NCHW, rFFT-based)
# -------------------------------

class AFNO2D(nn.Module):
    """
    AFNO-like mixer in NCHW layout.

    Ref: NVlabs AFNO-transformer.
    - rFFT2 on spatial dims.
    - Split channels into num_blocks groups; for each frequency location,
      apply complex MLP (two linear layers + GELU) along the channel dim.
    - Only low-frequency rectangle (modes) is updated; rest kept as identity.
    - Optional soft-shrinkage for sparsity.
    """

    def __init__(
        self,
        dim: int,
        num_blocks: int = 8,
        modes=None,  # (m1, m2) on (H, W//2+1); if None, use all
        fft_norm: str = "ortho",
        sparsity_thresh: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        assert dim % num_blocks == 0, "dim must be divisible by num_blocks"
        self.dim = dim
        self.nb = num_blocks
        self.bs = dim // num_blocks
        self.modes = modes  # (m1, m2) or None
        self.fft_norm = fft_norm
        self.sparsity_thresh = float(sparsity_thresh)
        self.residual = residual

        # complex linear weights: shape (nb, bs, bs)
        self.w1_real = nn.Parameter(torch.randn(self.nb, self.bs, self.bs) * 0.02)
        self.w1_imag = nn.Parameter(torch.randn(self.nb, self.bs, self.bs) * 0.02)
        self.b1_real = nn.Parameter(torch.zeros(self.nb, self.bs))
        self.b1_imag = nn.Parameter(torch.zeros(self.nb, self.bs))

        self.w2_real = nn.Parameter(torch.randn(self.nb, self.bs, self.bs) * 0.02)
        self.w2_imag = nn.Parameter(torch.randn(self.nb, self.bs, self.bs) * 0.02)
        self.b2_real = nn.Parameter(torch.zeros(self.nb, self.bs))
        self.b2_imag = nn.Parameter(torch.zeros(self.nb, self.bs))

        # 1x1 fuse after iFFT
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    def _complex_linear(
        self,
        z_real: torch.Tensor,
        z_imag: torch.Tensor,
        W_real: torch.Tensor,
        W_imag: torch.Tensor,
        b_real: torch.Tensor = None,
        b_imag: torch.Tensor = None,
    ):
        """
        z: (..., nb, bs)
        W: (nb, bs, bs)
        y = z @ W + b
        """
        y_real = torch.einsum("...ni,nij->...nj", z_real, W_real) - torch.einsum(
            "...ni,nij->...nj", z_imag, W_imag
        )
        y_imag = torch.einsum("...ni,nij->...nj", z_real, W_imag) + torch.einsum(
            "...ni,nij->...nj", z_imag, W_real
        )
        if b_real is not None:
            y_real = y_real + b_real
            y_imag = y_imag + b_imag
        return y_real, y_imag

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape
        assert C == self.dim

        # rFFT2 over spatial dims
        Z = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)  # (B,C,H,Wf)
        _, _, Hf, Wf = Z.shape

        Zr, Zi = Z.real, Z.imag  # (B,C,Hf,Wf)

        # reshape to groups: (B,C,Hf,Wf) -> (B,Hf,Wf,nb,bs)
        Zr = Zr.view(B, self.nb, self.bs, Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()
        Zi = Zi.view(B, self.nb, self.bs, Hf, Wf).permute(0, 3, 4, 1, 2).contiguous()

        # complex 2-layer MLP
        h_real, h_imag = self._complex_linear(
            Zr, Zi, self.w1_real, self.w1_imag, self.b1_real, self.b1_imag
        )
        h_real = F.gelu(h_real)
        h_imag = F.gelu(h_imag)
        y_real, y_imag = self._complex_linear(
            h_real, h_imag, self.w2_real, self.w2_imag, self.b2_real, self.b2_imag
        )

        # optional soft-shrinkage
        if self.sparsity_thresh > 0.0:
            y_real = F.softshrink(y_real, lambd=self.sparsity_thresh)
            y_imag = F.softshrink(y_imag, lambd=self.sparsity_thresh)

        # only update low-frequency modes if self.modes is not None
        if self.modes is not None:
            m1 = min(int(self.modes[0]), Hf)
            m2 = min(int(self.modes[1]), Wf)
            mask = torch.zeros(Hf, Wf, device=x.device, dtype=torch.bool)
            mask[:m1, :m2] = True
        else:
            mask = torch.ones(Hf, Wf, device=x.device, dtype=torch.bool)

        mask = mask.view(1, Hf, Wf, 1, 1)  # broadcast to (B,Hf,Wf,nb,bs)

        # combine updated low-freq with original high-freq
        Zr_out = torch.where(mask, y_real, Zr)
        Zi_out = torch.where(mask, y_imag, Zi)

        # back to (B,C,Hf,Wf)
        Zr_out = Zr_out.permute(0, 3, 4, 1, 2).contiguous().view(B, C, Hf, Wf)
        Zi_out = Zi_out.permute(0, 3, 4, 1, 2).contiguous().view(B, C, Hf, Wf)

        Z_out = torch.complex(Zr_out, Zi_out)

        y = torch.fft.irfft2(Z_out, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        y = self.fuse(y)
        return x + y if self.residual else y


# -------------------------------
# 3. FNO-style truncated spectral conv (rFFT, single low-freq block)
# -------------------------------

class SpectralConv2dTruncated(nn.Module):
    """
    FNO-style truncated spectral convolution in 2D (NCHW).
    Ref: Fourier Neural Operator.

    - 对空间维做 rFFT2。
    - 只在左上角低频矩形 (m1, m2) 上学习 Cin->Cout 的复权重。
    - 每个频率位置 (u, v) 对应一个 Cin×Cout 的复线性映射。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        modes=(16, 16),  # modes along (H, W_freq=W//2+1)
        fft_norm: str = "ortho",
        weight_scale=None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.m1 = int(modes[0])
        self.m2 = int(modes[1])
        self.fft_norm = fft_norm

        if weight_scale is None:
            weight_scale = 1.0 / (in_ch * out_ch) ** 0.5

        # real & imag weights: (Cin, Cout, m1, m2)
        self.wr = nn.Parameter(
            weight_scale * torch.randn(in_ch, out_ch, self.m1, self.m2)
        )
        self.wi = nn.Parameter(
            weight_scale * torch.randn(in_ch, out_ch, self.m1, self.m2)
        )

    def _get_weight(self, Hf: int, Wf: int):
        """
        Clip the learnable weights to the current frequency resolution.
        """
        m1 = min(self.m1, Hf)
        m2 = min(self.m2, Wf)
        Wr = self.wr[:, :, :m1, :m2]
        Wi = self.wi[:, :, :m1, :m2]
        return Wr, Wi, m1, m2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,Cin,H,W) -> (B,Cout,H,W)
        """
        B, Cin, H, W = x.shape
        assert Cin == self.in_ch, f"in_ch mismatch: {Cin} vs {self.in_ch}"

        # rFFT2 on spatial dims
        X = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)  # (B,Cin,H,Wf)
        _, _, Hf, Wf = X.shape

        Wr, Wi, m1, m2 = self._get_weight(Hf, Wf)

        # target spectral tensor
        Y = torch.zeros(
            B, self.out_ch, Hf, Wf, dtype=X.dtype, device=x.device
        )  # complex

        if m1 > 0 and m2 > 0:
            # low-frequency block: X_sub (B,Cin,m1,m2)
            X_sub = X[:, :, :m1, :m2]  # (B,Cin,m1,m2)

            # complex linear: (B,Cin,m1,m2) x (Cin,Cout,m1,m2) -> (B,Cout,m1,m2)
            # einsum indices: b c h w , c o h w -> b o h w
            Yr = torch.einsum(
                "bchq,cohq->bohq", X_sub.real, Wr
            ) - torch.einsum("bchq,cohq->bohq", X_sub.imag, Wi)
            Yi = torch.einsum(
                "bchq,cohq->bohq", X_sub.real, Wi
            ) + torch.einsum("bchq,cohq->bohq", X_sub.imag, Wr)

            Y_low = torch.complex(Yr, Yi)
            Y[:, :, :m1, :m2] = Y_low

        # inverse rFFT2 back to spatial
        y = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        return y


# -------------------------------
# 4. FFC-style global-local block
# -------------------------------

class FFCGlobalLocal2D(nn.Module):
    """
    FFC-like block (NCHW):
      - Split channels into local (1-ratio_g) and global (ratio_g).
      - Local branch: 3x3 conv or DWConv+PWConv.
      - Global branch: truncated FNO-style spectral conv (SpectralConv2dTruncated).
      - Concat & 1x1 fuse, with residual.
    """

    def __init__(
        self,
        dim: int,
        ratio_g: float = 0.5,
        modes=(16, 16),
        fft_norm: str = "ortho",
        residual: bool = True,
        local_dw: bool = False,
    ):
        super().__init__()
        assert 0.0 < ratio_g < 1.0
        self.dim = dim
        self.cg = int(round(dim * ratio_g))
        self.cl = dim - self.cg
        self.residual = residual

        if local_dw:
            self.local = nn.Sequential(
                nn.Conv2d(self.cl, self.cl, 3, padding=1, groups=self.cl, bias=False),
                nn.GELU(),
                nn.Conv2d(self.cl, self.cl, 1, bias=False),
            )
        else:
            self.local = nn.Conv2d(self.cl, self.cl, 3, padding=1, bias=False)

        # global spectral branch
        if self.cg > 0:
            self.global_spec = SpectralConv2dTruncated(
                self.cg, self.cg, modes=modes, fft_norm=fft_norm
            )
        else:
            self.global_spec = None

        self.fuse = nn.Conv2d(self.cl + self.cg, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C=dim, H, W)
        """
        if self.cg > 0:
            xl, xg = torch.split(x, [self.cl, self.cg], dim=1)
        else:
            xl, xg = x, None

        yl = self.local(xl)
        yg = self.global_spec(xg) if xg is not None else None

        if yg is not None:
            y = torch.cat([yl, yg], dim=1)
        else:
            y = yl

        y = self.fuse(y)
        return x + y if self.residual else y


# -------------------------------
# 5. DeformFNO2D (FNO with learned deformations)
# -------------------------------

class DeformFNO2D(nn.Module):
    """
    FNO with learned deformations (Geo-FNO style, NCHW):

      x --(flowNet)-> warp(x) --(SpectralConv2dTruncated)-> y_spec
        + 1x1(x) -> fuse -> y

    flowNet 预测 [-1,1]^2 归一化坐标系上的形变场 (dx, dy)，
    再通过 grid_sample 做可学习的坐标变换。
    """

    def __init__(
        self,
        dim: int,
        modes=(16, 16),
        flow_hidden: int = 32,
        flow_scale: float = 0.25,
        fft_norm: str = "ortho",
        residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.flow_scale = float(flow_scale)
        self.residual = residual

        # flow prediction network: depthwise 3x3 + pointwise 1x1
        self.flow = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.GELU(),
            nn.Conv2d(dim, flow_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(flow_hidden, 2, 1, bias=False),
            nn.Tanh(),  # outputs in [-1,1]
        )

        self.spec = SpectralConv2dTruncated(dim, dim, modes=modes, fft_norm=fft_norm)
        self.pw = nn.Conv2d(dim, dim, 1, bias=False)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    @staticmethod
    def _make_base_grid(B: int, H: int, W: int, device, dtype):
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2), (x,y)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        return grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape

        flow = self.flow(x) * self.flow_scale  # (B,2,H,W)
        base = self._make_base_grid(B, H, W, x.device, x.dtype)  # (B,H,W,2)
        grid = base + flow.permute(0, 2, 3, 1)  # (B,H,W,2), grid_sample expects (x,y)

        # warp in spatial domain
        x_warp = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

        # FNO on warped feature + local 1x1 on original
        y = self.spec(x_warp) + self.pw(x)
        y = self.fuse(y)
        return x + y if self.residual else y


# -------------------------------
# 6. WeightedFNO2D (WFNO-style)
# -------------------------------

class WeightedFNO2D(nn.Module):
    """
    WFNO-style weighted Fourier Neural Operator (DiffFNO-inspired, NCHW):

      - 先对 x 做一次低频 FNO 卷积 SpectralConv2dTruncated 得到 base。
      - 再把 base 映射到频域，对低频块 (m1,m2) 做输入自适应的幅度门控。

    对应 DiffFNO 里的 WFNO「mode rebalancing」思想：在频域对各个 mode 重新加权。
    """

    def __init__(
        self,
        dim: int,
        modes=(16, 16),
        gate_hidden: int = 128,
        fft_norm: str = "ortho",
        residual: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.m1 = int(modes[0])
        self.m2 = int(modes[1])
        self.fft_norm = fft_norm
        self.residual = residual

        # base FNO conv
        self.spec = SpectralConv2dTruncated(dim, dim, modes=modes, fft_norm=fft_norm)

        # global descriptor -> per-mode gate (shared across channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(dim, gate_hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(gate_hidden, self.m1 * self.m2, 1, bias=False),
        )

        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape
        assert C == self.dim

        # base FNO response in spatial domain
        base = self.spec(x)  # (B,C,H,W)

        # rFFT2 of the FNO output
        Z = torch.fft.rfft2(base, dim=(-2, -1), norm=self.fft_norm)  # (B,C,H,Wf)
        _, _, Hf, Wf = Z.shape

        # effective modes under current resolution
        m1 = min(self.m1, Hf)
        m2 = min(self.m2, Wf)

        if m1 > 0 and m2 > 0:
            # dynamic gate from input (global descriptor)
            g = self.pool(x)  # (B,C,1,1)
            g = self.gate(g)  # (B, m1*m2, 1, 1) (we will slice later)
            g = g.view(B, 1, self.m1, self.m2)
            g = g[:, :, :m1, :m2]  # (B,1,m1,m2)
            # stable amplitude scaling in roughly (0.5, 1.5)
            g = 1.0 + 0.5 * torch.tanh(g)

            # apply gating on low-frequency block
            Z_low = Z[:, :, :m1, :m2]  # (B,C,m1,m2)
            Z_low = Z_low * g  # broadcast along channels

            Z = Z.clone()
            Z[:, :, :m1, :m2] = Z_low

        # back to spatial
        y = torch.fft.irfft2(Z, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        y = self.fuse(y)
        return x + y if self.residual else y


# -------------------------------
# 7. FSEL-style frequency-spatial plug-in
# -------------------------------

class FSELPlugIn2D(nn.Module):
    """
    Frequency-Spatial Entanglement plug-in (lightweight, NCHW):

      - Spatial: DWConv(3x3) + PWConv
      - Spectral: rFFT2 -> 1x1 conv on real/imag -> SE-like gate on magnitude
                  -> 只调幅不改相 -> irFFT2
      - Fuse: concat [xs, yf] -> 1x1 -> residual

    抽象了 FSEL 里「频域 self-attn + 空间 FFN entanglement」的思想，
    但用更轻量的 CNN 形式实现，方便直接插到 CNN / U-Net backbone 里。
    """

    def __init__(
        self,
        dim: int,
        fft_norm: str = "ortho",
        spatial_dw: bool = True,
        residual: bool = True,
        gate_ratio: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.fft_norm = fft_norm
        self.residual = residual

        # Spatial branch
        if spatial_dw:
            self.spatial = nn.Sequential(
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, bias=False),
            )
        else:
            self.spatial = nn.Conv2d(dim, dim, 3, padding=1, bias=False)

        # Spectral branch: simple channel mixing in frequency domain
        self.freq_real = nn.Conv2d(dim, dim, 1, bias=False)
        self.freq_imag = nn.Conv2d(dim, dim, 1, bias=False)

        # magnitude SE gate (reduce -> expand)
        hidden = max(1, int(dim * gate_ratio))
        self.freq_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, bias=False),
            nn.Sigmoid(),
        )

        # Fuse spatial & spectral features
        self.fuse = nn.Conv2d(dim * 2, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape

        # spatial branch
        xs = self.spatial(x)

        # spectral branch
        Z = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)  # (B,C,H,Wf)
        Zr = self.freq_real(Z.real)
        Zi = self.freq_imag(Z.imag)
        Zm = torch.complex(Zr, Zi)

        # SE-gate on magnitude (per-channel)
        mag = torch.abs(Zm)              # (B,C,H,Wf)
        # 先在频率维上求均值，再用 AdaptiveAvgPool2d(1) 做全局 pooling
        mag_spatial = mag.mean(dim=-1, keepdim=True)  # (B,C,H,1)
        g = self.freq_gate(mag_spatial)  # (B,C,1,1) in (0,1)

        ang = torch.angle(Zm)
        scale = 0.5 + g  # (0.5, 1.5)
        Zg = torch.polar(mag * scale, ang)

        yf = torch.fft.irfft2(Zg, s=(H, W), dim=(-2, -1), norm=self.fft_norm)

        # fuse
        y = self.fuse(torch.cat([xs, yf], dim=1))
        return x + y if self.residual else y


# -------------------------------
# 8. SpectGating2D (SpectFormer-style)
# -------------------------------

class SpectGating2D(nn.Module):
    """
    SpectFormer-style pure spectral gating block (NCHW):

      - rFFT2 -> magnitude & phase
      - gate(magnitude) via a small CNN
      - amplitude scaling (1 + gate) with fixed phase
      - irFFT2 back to spatial + 1x1 fuse + residual
    """

    def __init__(
        self,
        dim: int,
        fft_norm: str = "ortho",
        residual: bool = True,
        gate_depthwise: bool = True,
        gate_hidden: int = 0,
    ):
        super().__init__()
        self.dim = dim
        self.fft_norm = fft_norm
        self.residual = residual

        layers = []
        if gate_depthwise:
            layers += [
                nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
                nn.GELU(),
            ]

        if gate_hidden and gate_hidden > 0:
            layers += [
                nn.Conv2d(dim, gate_hidden, 1, bias=False),
                nn.GELU(),
                nn.Conv2d(gate_hidden, dim, 1, bias=False),
            ]
        else:
            layers += [nn.Conv2d(dim, dim, 1, bias=False)]

        self.gater = nn.Sequential(*layers)
        self.fuse = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,C,H,W)
        """
        B, C, H, W = x.shape

        Z = torch.fft.rfft2(x, dim=(-2, -1), norm=self.fft_norm)  # (B,C,H,Wf)
        mag, ang = torch.abs(Z), torch.angle(Z)

        # gate on magnitude
        gate = torch.sigmoid(self.gater(mag))  # (B,C,H,Wf) in (0,1)
        # amplitude scaling in roughly (1, 2)
        Z_new = torch.polar(mag * (1.0 + gate), ang)

        y = torch.fft.irfft2(Z_new, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        y = self.fuse(y)
        return x + y if self.residual else y

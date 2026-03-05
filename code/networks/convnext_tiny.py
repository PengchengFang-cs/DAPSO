import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from networks.global_mixer import GFNetMixer2D, AFNO2D, FFCGlobalLocal2D, DeformFNO2D, WeightedFNO2D, FSELPlugIn2D, SpectGating2D, SpectralConv2dTruncated
from networks.DAPSO_test_cur_best import DAPSO

def build_dapso(hidden_dim, **overrides):
    base_cfg = dict(
        dim=hidden_dim,
        rank=8,
        band=None,
        basis='identity',
        local='conv',
        axis='hc_wc',
        h_gating_mode='discrete',
        w_gating_mode='discrete',
        disc_bins=96,
        basis_h='learned_ortho',
        basis_w='identity',
        group_h=None,
        group_w=None,
    )
    base_cfg.update(overrides)

    print("[DAPSO config]")
    for k, v in base_cfg.items():
        print(f"  {k}: {v}")

    return DAPSO(**base_cfg)

def build_spectral_mixer(kind: str, dim: int, **kw):
    """
    根据 kind 创建相应的频域模块实例。

    参数：
      kind: 'dapso', 'gfnet', 'afno', 'ffc', 'deformfno', 'fno',
            'wfno', 'fsel', 'spectformer'
      dim:  输入和输出的通道数
      **kw: 可以覆盖的超参
    """
    kind = kind.lower()

    # 1. DAPSO（保持您已有的实现）
    if kind == 'dapso':
        return build_dapso(dim, **kw)

    # GFNet-style 全局滤波
    elif kind == 'gfnet':
        return GFNetMixer2D(
            dim=dim,
            bins=kw.get('bins', (64, 64)),      # 粗频网格大小
            fft_norm=kw.get('fft_norm', 'ortho'),
            residual=kw.get('residual', True),
            use_bias=kw.get('use_bias', False),
        )

    # 2. AFNO-style token mixer
    elif kind == 'afno':
        return AFNO2D(
            dim=dim,
            num_blocks=kw.get('num_blocks', 8),
            modes=kw.get('modes', None),        # None 表示不裁剪频率
            fft_norm=kw.get('fft_norm', 'ortho'),
            sparsity_thresh=kw.get('sparsity_thresh', 0.01),
            residual=kw.get('residual', True),
        )

    # 3. FFC-style 局部+全局
    elif kind == 'ffc':
        return FFCGlobalLocal2D(
            dim=dim,
            ratio_g=kw.get('ratio_g', 0.5),     # 全局通道比例
            modes=kw.get('modes', (16, 16)),
            fft_norm=kw.get('fft_norm', 'ortho'),
            residual=kw.get('residual', True),
            local_dw=kw.get('local_dw', True),  # 使用 DWConv+PWConv
        )

    # 4. FNO with learned deformations（Geo-FNO 风格）
    elif kind == 'deformfno':
        return DeformFNO2D(
            dim=dim,
            modes=kw.get('modes', (16, 16)),
            flow_hidden=kw.get('flow_hidden', 32),
            flow_scale=kw.get('flow_scale', 0.25),
            fft_norm=kw.get('fft_norm', 'ortho'),
            residual=kw.get('residual', True),
        )

    # 5. 标准 FNO 块（不带形变、不带门控）
    elif kind == 'fno':
        modes = kw.get('modes', (16, 16))
        fft_norm = kw.get('fft_norm', 'ortho')
        residual = kw.get('residual', True)

        class FNOBlock(nn.Module):
            def __init__(self, dim, modes, fft_norm, residual):
                super().__init__()
                self.spec = SpectralConv2dTruncated(dim, dim, modes=modes, fft_norm=fft_norm)
                self.pw = nn.Conv2d(dim, dim, 1, bias=False)
                self.residual = residual
            def forward(self, x):
                y = self.spec(x) + self.pw(x)
                return x + y if self.residual else y

        return FNOBlock(dim, modes, fft_norm, residual)

    # 6. Weighted FNO（DiffFNO/WFNO 风格）
    elif kind == 'wfno':
        return WeightedFNO2D(
            dim=dim,
            modes=kw.get('modes', (16, 16)),
            gate_hidden=kw.get('gate_hidden', 128),
            fft_norm=kw.get('fft_norm', 'ortho'),
            residual=kw.get('residual', True),
        )

    # 7. FSEL 风格频空增强
    elif kind == 'fsel':
        return FSELPlugIn2D(
            dim=dim,
            fft_norm=kw.get('fft_norm', 'ortho'),
            spatial_dw=kw.get('spatial_dw', True),  # True：DWConv+PWConv
            residual=kw.get('residual', True),
            gate_ratio=kw.get('gate_ratio', 0.5),
        )

    # 8. SpectFormer 风格纯频域 gating
    elif kind == 'spectformer':
        return SpectGating2D(
            dim=dim,
            fft_norm=kw.get('fft_norm', 'ortho'),
            residual=kw.get('residual', True),
            gate_depthwise=kw.get('gate_depthwise', True),
            gate_hidden=kw.get('gate_hidden', 0),
        )
    elif kind == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Unknown spectral mixer kind: {kind}")


class LayerNorm2d(nn.Module):
    """channels_first 的 LayerNorm，和你之前那个类似。"""
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps

    def forward(self, x):
        # x: (B,C,H,W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)  # channels_last
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim))
        ) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mixers = build_spectral_mixer('identity', dim=dim)  # kind: 'dapso', 'gfnet', 'afno', 'ffc', 'deformfno', 'fno', 'wfno', 'fsel', 'spectformer', 'identity'

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)              # (B,C,H,W)
        x = self.mixers(x)          # 频域混合模块
        x = x.permute(0, 2, 3, 1)       # -> (B,H,W,C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)       # -> (B,C,H,W)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXtTinyEncoder(nn.Module):
    """
    ConvNeXt Tiny encoder，返回 4 个 stage 的 feature：
    f1: 1/4,  f2: 1/8,  f3: 1/16,  f4: 1/32 分辨率
    """
    def __init__(self, in_chans=2, depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    Block(dim=dims[i],
                          drop_path=dp_rates[cur + j],
                          layer_scale_init_value=layer_scale_init_value)
                )
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]

        self.out_dims = dims

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # x: (B,in_chans,H,W), e.g. (B,2,256,256)
        feats = []

        # stage 0: stem -> stage0
        x = self.downsample_layers[0](x)  # (B,96,64,64)
        x = self.stages[0](x)
        feats.append(x)                   # f1: 64x64

        # stage 1
        x = self.downsample_layers[1](x)  # (B,192,32,32)
        x = self.stages[1](x)
        feats.append(x)                   # f2: 32x32

        # stage 2
        x = self.downsample_layers[2](x)  # (B,384,16,16)
        x = self.stages[2](x)
        feats.append(x)                   # f3: 16x16

        # stage 3
        x = self.downsample_layers[3](x)  # (B,768,8,8)
        x = self.stages[3](x)
        feats.append(x)                   # f4: 8x8

        # 返回 list: [f1,f2,f3,f4]
        return feats

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act2 = nn.GELU()

    def forward(self, x, skip):
        # x: (B,in_ch,H,W), skip: (B,skip_ch,H,W)
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        return x


class ConvNeXtTinyRecon(nn.Module):
    """
    ConvNeXt Tiny + U-Net 风格 decoder 做重建。

    - in_chans: 输入图像通道，比如 cc359 复数 = 2
    - out_chans: 输出图像通道，比如 cc359 复数 = 2
    - img_size: 输入 / 输出空间分辨率，比如 256
    """
    def __init__(self, in_chans=2, out_chans=2, img_size=256):
        super().__init__()
        self.img_size = img_size
        self.encoder = ConvNeXtTinyEncoder(in_chans=in_chans)

        dims = self.encoder.out_dims  # (96,192,384,768)
        # decoder 通道数可以根据需要调小点
        self.up4 = nn.Conv2d(dims[3], dims[2], kernel_size=1)  # 8->16 前的通道变换
        self.dec3 = UpBlock(dims[2], dims[2], dims[2])         # f4(upsampled) + f3 -> 384

        self.up3 = nn.Conv2d(dims[2], dims[1], kernel_size=1)
        self.dec2 = UpBlock(dims[1], dims[1], dims[1])         # -> 192

        self.up2 = nn.Conv2d(dims[1], dims[0], kernel_size=1)
        self.dec1 = UpBlock(dims[0], dims[0], dims[0])         # -> 96

        # 从 64x64 -> 256x256（x4）
        self.final_conv = nn.Sequential(
            nn.Conv2d(dims[0], dims[0] // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, out_chans, 3, padding=1),
        )

    def forward(self, x, k=None, q=None, **kwargs):
        """
        x: (B,in_chans,256,256)
        return: (B,out_chans,256,256)
        """
        f1, f2, f3, f4 = self.encoder(x)   # 64x64,32x32,16x16,8x8

        # 8x8 -> 16x16
        x4 = F.interpolate(f4, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = self.up4(x4)
        x3 = self.dec3(x4, f3)             # 16x16

        # 16x16 -> 32x32
        x3_up = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x3_up = self.up3(x3_up)
        x2 = self.dec2(x3_up, f2)          # 32x32

        # 32x32 -> 64x64
        x2_up = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
        x2_up = self.up2(x2_up)
        x1 = self.dec1(x2_up, f1)          # 64x64

        # 64x64 -> 256x256
        x_final = F.interpolate(x1, scale_factor=4, mode='bilinear', align_corners=False)
        x_final = self.final_conv(x_final)  # (B,out_chans,256,256)

        return x_final
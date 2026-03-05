# -*- coding: utf-8 -*-
# FNO-2D (ICLR 2021) faithful structure
# Ref idea: Li et al., "Fourier Neural Operator for Parametric PDEs" (ICLR 2021)

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Spectral Convolution (2D)
# ----------------------------
class SpectralConv2d(nn.Module):
    """
    2D spectral convolution used in FNO:
      - rFFT2 on (H, W), keep half-spectrum on W
      - learn complex weights on top modes1 x modes2 (top positive freq) and bottom modes1 x modes2 (negative freq)
      - inverse rFFT2 back to spatial
    Input shape : (B, C_in, H, W)   (real)
    Output shape: (B, C_out, H, W)  (real)
    """
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of modes along H
        self.modes2 = modes2  # number of modes along W (half-spectrum)

        # Complex weights, parameterized by (real, imag)
        # shape: (C_in, C_out, modes1, modes2)
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, 2))

    @staticmethod
    def compl_mul2d(a, w):
        """
        Complex multiply:
          a: (B, C_in, M1, M2) complex
          w: (C_in, C_out, M1, M2, 2) real -> (real, imag)
          returns: (B, C_out, M1, M2) complex
        """
        a_real, a_imag = a.real, a.imag
        w_real, w_imag = w[..., 0], w[..., 1]
        # (B, Cin, M1, M2) x (Cin, Cout, M1, M2) -> (B, Cout, M1, M2)
        out_real = torch.einsum('bixy,ioxy->boxy', a_real, w_real) - torch.einsum('bixy,ioxy->boxy', a_imag, w_imag)
        out_imag = torch.einsum('bixy,ioxy->boxy', a_real, w_imag) + torch.einsum('bixy,ioxy->boxy', a_imag, w_real)
        return torch.complex(out_real, out_imag)

    def forward(self, x):
        """
        x: (B, C_in, H, W) real
        """
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1) complex

        out_ft = x_ft.new_zeros((B, self.out_channels, H, W // 2 + 1))

        m1 = min(self.modes1, H)
        m2 = min(self.modes2, W // 2 + 1)

        # Top block (positive frequencies along H): 0 : m1
        if m1 > 0 and m2 > 0:
            out_ft[:, :, :m1, :m2] = self.compl_mul2d(
                x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2]
            )

        # Bottom block (negative frequencies along H): -m1 :
        if m1 > 0 and m2 > 0:
            out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
                x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2]
            )

        x = torch.fft.irfft2(out_ft, s=(H, W), dim=(-2, -1))  # (B, C_out, H, W) real
        return x


class FourierLocalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, modes1=16, modes2=16, local_k=3):
        super().__init__()
        self.global_fft = SpectralConv2d(in_ch, out_ch, modes1, modes2)
        self.local_conv = nn.Conv2d(in_ch, out_ch, local_k, padding=local_k//2)
        self.fuse = nn.Conv2d(out_ch, out_ch, 1)  # optional
        self.act = nn.GELU()

    def forward(self, x):
        y_global = self.global_fft(x)
        y_local = self.local_conv(x)
        return self.act(self.fuse(y_global + y_local))

# ----------------------------
# FNO-2D network
# ----------------------------
class FNO2d(nn.Module):
    """
    Faithful FNO-2D:
      u (B, C_in, H, W)
      + grid (x,y)  -> concat on channel-last
      -> fc0 (lift to width)
      -> 4 × [ SpectralConv2d(width->width) + 1×1 conv (w) + GELU ]
      -> fc1 (width->128) -> GELU -> fc2 (128->C_out)

    Args:
      modes1, modes2: number of Fourier modes kept in H/W
      width: channel width inside the Fourier layers
      in_channels: input channels (without grid)
      out_channels: output channels
      pad: optional spatial zero padding (for nonperiodic cases, typical 0 or 9)
    """
    def __init__(self, modes1, modes2, width, in_channels, out_channels, pad=0):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad = pad

        # lifting: (in_channels + 2 coords) -> width
        self.fc0 = nn.Linear(in_channels + 2, width)

        # 4 Fourier layers
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        # 1×1 conv shortcuts (time-domain linear)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        # projection head
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)

    @staticmethod
    def get_grid(B, H, W, device):
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        grid = torch.stack((yy, xx), dim=-1)  # (H, W, 2)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B, H, W, 2)
        return grid

    def forward(self, x):
        """
        x: (B, C_in, H, W) real
        return: (B, C_out, H, W)
        """
        B, Cin, H, W = x.shape
        device = x.device

        # optional zero-padding on spatial dims (nonperiodic)
        if self.pad and self.pad > 0:
            x = F.pad(x, (0, self.pad, 0, self.pad))  # pad W, then H
            H_pad, W_pad = x.shape[-2], x.shape[-1]
        else:
            H_pad, W_pad = H, W

        # to channel-last for fc0
        x_ = x.permute(0, 2, 3, 1)  # (B, H, W, Cin)
        grid = self.get_grid(B, H_pad, W_pad, device)  # match spatial after padding
        x_ = torch.cat((x_, grid), dim=-1)             # (B, H, W, Cin+2)
        x_ = self.fc0(x_)                              # (B, H, W, width)
        x_ = x_.permute(0, 3, 1, 2)                    # (B, width, H, W)

        # 4 Fourier layers with 1x1 residual
        x1 = self.conv0(x_) + self.w0(x_); x1 = F.gelu(x1)
        x2 = self.conv1(x1) + self.w1(x1); x2 = F.gelu(x2)
        x3 = self.conv2(x2) + self.w2(x2); x3 = F.gelu(x3)
        x4 = self.conv3(x3) + self.w3(x3); x4 = F.gelu(x4)

        # back to channel-last and project
        x4 = x4.permute(0, 2, 3, 1)         # (B, H, W, width)
        x4 = self.fc1(x4); x4 = F.gelu(x4)  # (B, H, W, 128)
        x4 = self.fc2(x4)                   # (B, H, W, Cout)
        x4 = x4.permute(0, 3, 1, 2)         # (B, Cout, H, W)

        # crop back if padded
        if self.pad and self.pad > 0:
            x4 = x4[..., :H, :W]

        return x4

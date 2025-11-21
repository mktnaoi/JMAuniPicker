import torch
import torch.nn as nn
import torch.nn.functional as F
from seisbench.models.base import WaveformModel


def interpolate_1d(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """1D linear interpolation with MPS compatibility."""
    if x.ndim != 3:
        raise ValueError("Input must be 3D (batch, channels, samples)")
    x = x.unsqueeze(2)
    x = F.interpolate(x, scale_factor=(1, scale_factor), mode="bilinear", align_corners=True)
    return x.squeeze(2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, kernel_size=3, stride=1, padding=1, dif_residual=False):
        super().__init__()
        self.identity = stride == 1 and in_channels == out_channels
        self.dif_residual = dif_residual

        hidden_dim = in_channels * expansion_factor
        self.layer1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.layer2 = nn.Sequential(nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=True), nn.GELU())
        self.layer3 = nn.Sequential(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=True))
        self.shortcut = nn.Identity() if self.identity else nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, normalized_shape=(x.shape[2],), eps=1e-3)
        x = x.permute(0, 2, 1)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.identity or self.dif_residual:
            x += self.shortcut(residual)
        return x


class SepConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class UpSepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, padding):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = SepConv1D(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channels, eps=1e-3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = interpolate_1d(x, scale_factor=self.scale_factor)
        x = self.relu(self.bn(self.conv(x)))
        return x


class ASPPModule(nn.Module):
    def __init__(self, bottom_channels, middle_channels, kernel_size, padding, dilation, sepconv=False):
        super().__init__()
        if sepconv:
            conv = SepConv1D(bottom_channels, middle_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        else:
            conv = nn.Conv1d(bottom_channels, middle_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
        self.atrous_conv = conv
        self.bn = nn.BatchNorm1d(middle_channels, eps=1e-3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPP(nn.Module):
    def __init__(self, bottom_channels, middle_channels, sepconv=False):
        super().__init__()
        self.aspp1 = ASPPModule(bottom_channels, middle_channels, 1, padding=0, dilation=1, sepconv=sepconv)
        self.aspp2 = ASPPModule(bottom_channels, middle_channels, 3, padding=6, dilation=6, sepconv=sepconv)
        self.aspp3 = ASPPModule(bottom_channels, middle_channels, 3, padding=12, dilation=12, sepconv=sepconv)
        self.aspp4 = ASPPModule(bottom_channels, middle_channels, 3, padding=18, dilation=18, sepconv=sepconv)
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(bottom_channels, middle_channels, 1, stride=1, bias=False),
            nn.BatchNorm1d(middle_channels, eps=1e-3),
            nn.ReLU(),
        )
        self.conv = nn.Conv1d(middle_channels * 5, middle_channels, 1, bias=False)
        self.bn = nn.BatchNorm1d(middle_channels, eps=1e-3)
        self.relu = nn.ReLU()

    def forward(self, x, mapsize):
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = self.global_avg_pool(x)
        aspp5 = interpolate_1d(aspp5, scale_factor=mapsize)
        out = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class UseModel(WaveformModel):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        nch_l1 = 8
        nch_l2 = 16
        nch_l3 = 32
        nch_l4 = 64
        nch_l5 = 128
        base_kernel = 7

        self.Level1_1 = ConvNeXtBlock(in_channels, nch_l1, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)
        self.Level1_2 = ConvNeXtBlock(nch_l1, nch_l1, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)

        self.Level2_1 = ConvNeXtBlock(nch_l1, nch_l2, expansion_factor=4, kernel_size=base_kernel, stride=2, padding=3, dif_residual=True)
        self.Level2_2 = ConvNeXtBlock(nch_l2, nch_l2, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)

        self.Level3_1 = ConvNeXtBlock(nch_l2, nch_l3, expansion_factor=4, kernel_size=base_kernel, stride=2, padding=3, dif_residual=True)
        self.Level3_2 = ConvNeXtBlock(nch_l3, nch_l3, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)

        self.Level4_1 = ConvNeXtBlock(nch_l3, nch_l4, expansion_factor=4, kernel_size=base_kernel, stride=2, padding=3, dif_residual=True)
        self.Level4_2 = ConvNeXtBlock(nch_l4, nch_l4, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)

        self.Level5_1 = ConvNeXtBlock(nch_l4, nch_l5, expansion_factor=4, kernel_size=base_kernel, stride=2, padding=3, dif_residual=True)
        self.Level5_2 = ConvNeXtBlock(nch_l5, nch_l5, expansion_factor=4, kernel_size=base_kernel, stride=1, padding=3, dif_residual=True)

        self.ASPP = ASPP(nch_l5, nch_l3, sepconv=True)

        self.Level3_3 = SepConv1D(nch_l3 * 2, nch_l3, kernel_size=base_kernel, stride=1, padding=3, bias=False)
        self.Level3_4 = SepConv1D(nch_l3, nch_l3, kernel_size=base_kernel, stride=1, padding=3, bias=False)

        self.Level2_3 = UpSepLayer(nch_l3, nch_l2, scale_factor=2, kernel_size=base_kernel, padding=3)
        self.Level2_4 = SepConv1D(nch_l2, nch_l2, kernel_size=base_kernel, stride=1, padding=3, bias=False)

        self.Level1_3 = UpSepLayer(nch_l2, nch_l1, scale_factor=2, kernel_size=base_kernel, padding=3)
        self.Level1_4 = SepConv1D(nch_l1, nch_l1, kernel_size=base_kernel, stride=1, padding=3, bias=False)

        self.OutConv = nn.Conv1d(nch_l1, 3, 1, padding=0)

    def forward(self, x):
        x = self.Level1_1(x)
        x = self.Level1_2(x)
        x1 = x

        x = self.Level2_1(x)
        x = self.Level2_2(x)
        x2 = x

        x = self.Level3_1(x)
        x = self.Level3_2(x)
        x3 = x

        x = self.Level4_1(x)
        x = self.Level4_2(x)

        x = self.Level5_1(x)
        x = self.Level5_2(x)
        x = self.ASPP(x, mapsize=x.shape[2])

        x = interpolate_1d(x, scale_factor=4)
        x = torch.cat((x3, x), dim=1)
        x = self.Level3_3(x)
        x = self.Level3_4(x)

        x = self.Level2_3(x)
        x = self.Level2_4(x)

        x = self.Level1_3(x)
        x = self.Level1_4(x)
        x = self.OutConv(x)

        return x


__all__ = ["UseModel"]

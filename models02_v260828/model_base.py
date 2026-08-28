import torch
import torch.nn as nn
import torch.nn.functional as F
from seisbench.models.base import WaveformModel
# MBP: source /Users/naoi/python_env/seisbench/bin/activate
# python org_models.py


# Model 01: Original PhaseNet -------------------------------------------
# Convolution19個（ConvTransposeを４つ含む）
# 簡単のため，サンプルサイズをn^2に固定
# 損失関数には，torch.nn.CrossEntropyLossを用いることを想定して最終レイヤのsoftmaxを除いている．

class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, kernel_size=3, stride=1, padding=1, dif_residual=False):       
        super().__init__()
        
        # dif_residual = True : n_channels≠out_channelsやstride > 1の場合にも残差接続を行う
        # self.identity = Trueの場合は恒等写像 
        self.identity = (stride == 1) and (in_channels == out_channels)
        self.dif_residual = dif_residual
        
        hidden_dim = in_channels * expansion_factor
        # Depthwise Convolution
        self.layer1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                padding=padding, groups=in_channels, bias=False)        
        # Expantion (1x1 convolution)
        self.layer2 = nn.Sequential(nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=True),                                    
                                    nn.GELU())
        # Projection (1x1 convolution)
        self.layer3 = nn.Sequential(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=True))
        
        if self.identity:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                           
    def forward(self, x):
        residual = x
        
        x = self.layer1(x)
        
        # Layer Normalization
        x = x.permute(0, 2, 1)
        x = F.layer_norm(x, normalized_shape=(x.shape[2],), eps=1e-3)
        x = x.permute(0, 2, 1)

        x = self.layer2(x) 
        x = self.layer3(x)  
        
       # Add shortcut if applicable
        if self.identity or self.dif_residual:
            x += self.shortcut(residual)
        return x

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4, kernel_size=3, stride=1, padding=1, dif_residual=False):
        super().__init__()
        self.identity = (stride == 1) and (in_channels == out_channels)
        self.dif_residual = dif_residual
        
        hidden_dim = in_channels * expansion_factor
        self.expand = nn.Sequential(nn.Conv1d(in_channels, hidden_dim, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(hidden_dim, eps=1e-3),
                                    nn.ReLU6(inplace=True))
        # Depthwise Convolution
        self.depthwise = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                    padding=padding, groups=hidden_dim, bias=False),
                                    nn.BatchNorm1d(hidden_dim, eps=1e-3),
                                    nn.ReLU6(inplace=True))        
        # Projection (1x1 convolution)
        self.project = nn.Sequential(nn.Conv1d(hidden_dim, out_channels, kernel_size=1, bias=False),
                                     nn.BatchNorm1d(out_channels, eps=1e-3))
        
        if self.identity:
            self.shortcut = nn.Identity()
        else:
            print('!--------!!!!!!!!!')
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                          
    def forward(self, x):
        residual = x       
        x = self.expand(x) 
        x = self.depthwise(x) 
        x = self.project(x)  
        
       # Add shortcut if applicable
        if self.identity or self.dif_residual:
            x += self.shortcut(residual)
        return x

class NormalSepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.conv  = SepConv1D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)        
        self.bn    = nn.BatchNorm1d(out_channels, eps=1e-3)
        self.relu  = nn.ReLU()              
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))         
        return x
                
class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.conv  = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)        
        self.bn    = nn.BatchNorm1d(out_channels, eps=1e-3)
        self.relu  = nn.ReLU()              
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))         
        return x

# class DownLayer(nn.Module):
        
#     def __init__(self, channels, kernel_size, stride, padding):
#         super().__init__()
#         self.conv  = ConvNeXt(channels, channels, expansion_factor=4, kernel_size=kernel_size, stride=stride, padding=padding)
#         # self.conv  = SepConv1D(channels, channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=1, bias=False)                
#         self.bn    = nn.BatchNorm1d(channels, eps=1e-3)
#         self.relu  = nn.ReLU() 
            
#     def forward(self, x):
#         x = self.relu(self.bn(self.conv(x)))       
#         return x


def interpolate_1D(x: torch.Tensor, scale_factor: float) -> torch.Tensor:
    """
    MPS does not support the 1D version of F.interpolate(x, scale_factor=..., mode="linear"), 
    so the 2D version of the function is used instead.
    """
    if x.ndim != 3:
        raise ValueError("Input must be 3D (B, C, L)")
    # (B, C, L) → (B, C, 1, L)
    x = x.unsqueeze(2)
    # interpolate last dim (width)
    x = F.interpolate(x, scale_factor=(1, scale_factor), mode="bilinear", align_corners=True)
    # (B, C, 1, L*) → (B, C, L*)
    return x.squeeze(2)


class UpLayer(nn.Module):
    # 250220修正．BNとReLUの重複を削除        
    def __init__(self, in_channels, out_channels, scale_factor, kernel_size, padding):     
        super().__init__() 
        self.scale_factor = scale_factor
        self.conv     = NormalConv(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=1, bias=False)           
    def forward(self, x):
        x = self.conv(interpolate_1D(x, scale_factor=self.scale_factor))
        return x


class SepConv1D(nn.Module):
    """
    Depthwise Separable Convolution for 1D data
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, dilation=dilation, bias=False)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
        
class ASPPModule(nn.Module):
    def __init__(self, bottom_channels, middle_channels, kernel_size, padding, dilation):
        super().__init__()
        self.atrous_conv = SepConv1D(bottom_channels, middle_channels, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False) 
        self.bn = nn.BatchNorm1d(middle_channels, eps=1e-3)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.bn(self.atrous_conv(x)))        
        return x

class ASPP(nn.Module):
    def __init__(self, bottom_channels, middle_channels):
        super().__init__()
        self.aspp1 = ASPPModule(bottom_channels, middle_channels, 1, padding=0, dilation=1)
        self.aspp2 = ASPPModule(bottom_channels, middle_channels, 3, padding=6, dilation=6)
        self.aspp3 = ASPPModule(bottom_channels, middle_channels, 3, padding=12, dilation=12)
        self.aspp4 = ASPPModule(bottom_channels, middle_channels, 3, padding=18, dilation=18)
        # self.aspp1 = ASPPModule(128, 16)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                        nn.Conv1d(bottom_channels, middle_channels, 1, stride=1, bias=False),
                                        nn.BatchNorm1d(middle_channels, eps=1e-3),
                                        nn.ReLU())
        # SepConvよりConv1dのほうがパラメタ数少ない
        self.conv = nn.Conv1d(middle_channels*5, middle_channels, 1, bias=False)
        self.bn   = nn.BatchNorm1d(middle_channels, eps=1e-3)
        self.relu = nn.ReLU()
                
    def forward(self, x, mapsize):
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = self.global_avg_pool(x)
        aspp5 = F.interpolate(aspp5, size=mapsize, mode='linear', align_corners=True)
        out = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        out = self.relu(self.bn(self.conv(out)))              
        return out
            
class UseModel(WaveformModel):
    def __init__(self, in_channels=3, **kwargs):
        super().__init__(**kwargs)
        
        nch_L1 = 8   # original length
        nch_L2 = 16  # 1/2
        nch_L3 = 32  # 1/4
        nch_L4 = 64  # 1/8
        nch_L5 = 128 # 1/16
        base_kernel_size = 7
        
        
        # 出力値が各LvのCh数になっているかで判定
        # Level 1 
        self.Level1_1 = NormalConv(in_channels, out_channels=nch_L1, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)        
        self.Level1_2 = NormalConv(in_channels=nch_L1, out_channels=nch_L1, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
         
        # Level 2     
        self.Level2_1 = NormalConv(in_channels=nch_L1, out_channels=nch_L2, kernel_size=base_kernel_size, stride=4, padding=3, bias=False)            
        self.Level2_2 = NormalConv(in_channels=nch_L2, out_channels=nch_L2, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
                          
        # Level 3 
        self.Level3_1 = NormalConv(in_channels=nch_L2, out_channels=nch_L3, kernel_size=base_kernel_size, stride=4, padding=3, bias=False)            
        self.Level3_2 = NormalConv(in_channels=nch_L3, out_channels=nch_L3, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
                
        # Level 4
        self.Level4_1 = NormalConv(in_channels=nch_L3, out_channels=nch_L4, kernel_size=base_kernel_size, stride=4, padding=3, bias=False)            
        self.Level4_2 = NormalConv(in_channels=nch_L4, out_channels=nch_L4, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
        
        # Level 5
        self.Level5_1 = NormalConv(in_channels=nch_L4, out_channels=nch_L5, kernel_size=base_kernel_size, stride=4, padding=3, bias=False)            
        self.Level5_2 = NormalConv(in_channels=nch_L5, out_channels=nch_L5, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
                
        # Level 4
        self.Level4_3 = UpLayer(nch_L5, nch_L4, scale_factor=4, kernel_size=base_kernel_size, padding=3)
        self.Level4_4 = NormalConv(in_channels=nch_L4*2, out_channels=nch_L4, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
        
        # Level 3
        self.Level3_3 = UpLayer(nch_L4, nch_L3, scale_factor=4, kernel_size=base_kernel_size, padding=3)
        self.Level3_4 = NormalConv(in_channels=nch_L3*2, out_channels=nch_L3, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
        
        # Level 2
        self.Level2_3 = UpLayer(nch_L3, nch_L2, scale_factor=4, kernel_size=base_kernel_size, padding=3)
        self.Level2_4 = NormalConv(in_channels=nch_L2*2, out_channels=nch_L2, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
        
        # Level 1
        self.Level1_3 = UpLayer(nch_L2, nch_L1, scale_factor=4, kernel_size=base_kernel_size, padding=3)
        self.Level1_4 = NormalConv(in_channels=nch_L1*2, out_channels=nch_L1, kernel_size=base_kernel_size, stride=1, padding=3, bias=False)
                                             
        self.OutConv  = nn.Conv1d(nch_L1, 3, 1, padding=0)
    

    def forward(self, x):
        
        # Level 1    
        x = self.Level1_1(x)
        x = self.Level1_2(x)
        x1 = x
        
        # Level 2 (size 1/2)        
        x = self.Level2_1(x)
        x = self.Level2_2(x)
        x2 = x
        
        # Level 3 (size 1/4)
        x = self.Level3_1(x)
        x = self.Level3_2(x)        
        x3 = x
        
        
        # Level 4 (size 1/8)
        x = self.Level4_1(x)
        x = self.Level4_2(x)
        x4 = x
        
        # Level 5 (size 1/16)
        x = self.Level5_1(x)
        x = self.Level5_2(x)
        
        # Level 4
        x = self.Level4_3(x)
        x = torch.cat([x4, x], dim=1) 
        x = self.Level4_4(x)

        # Level 3 (size 1/4)
        x = self.Level3_3(x)
        x = torch.cat([x3, x], dim=1) 
        x = self.Level3_4(x)        
        
        # Level 2 (size 1/2)        
        x = self.Level2_3(x)
        x = torch.cat([x2, x], dim=1) 
        x = self.Level2_4(x)
        
        # Level 1
        x = self.Level1_3(x) 
        x = torch.cat([x1, x], dim=1) 
        x = self.Level1_4(x)        
        x = self.OutConv(x) 
    
        return x
    
if __name__ == "__main__":
    from torchinfo import summary

    # model summaryの表示
    in_channels = 3
    n_samples = 4096
    
    model = UseModel(in_channels=in_channels)
    
    # Block debug ----
    # model = ConvNeXtBlock(in_channels=3, out_channels=3, expansion_factor=4, kernel_size=3, stride=1, padding=1, dif_residual=False)
    # model = InvertedResidualBlock(in_channels=3, out_channels=3, expansion_factor=4, kernel_size=3, stride=2, padding=1, dif_residual=True)
    # model = NormalSepConv(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=0, bias=False)
    # model = NormalConv(in_channels=3, out_channels=3, kernel_size=3, stride=2, padding=1, bias=False)

    # model = UseModel(in_channels=in_channels)
    summary(model, input_size=(1, in_channels, n_samples))
    

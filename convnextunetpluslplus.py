import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
import convnext
from typing import Sequence, List, Union, Optional, Dict, Any
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder
from segmentation_models_pytorch.decoders.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.decoders.pan.decoder import PANDecoder

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# class DecoderBlock(nn.Module):
#     """使用反卷积的U-Net++解码器块"""

#     def __init__(
#             self,
#             in_channels: int,
#             skip_channels: int,
#             out_channels: int,
#             activation: nn.Module = nn.ReLU(inplace=True),
#             dropout: float = 0.1,
#             # 反卷积参数
#             kernel_size: int = 4,  # 反卷积核大小，通常设为4以配合stride=2实现2倍上采样
#             stride: int = 2,       # 步长=2实现2倍放大
#             padding: int = 1       # 填充=1保持边缘对齐
#     ):
#         super().__init__()
        
#         # 反卷积层：同时实现上采样（2倍）和通道匹配
#         # 反卷积公式：output_size = (input_size - 1) * stride + kernel_size - 2*padding
#         # 当 input_size 为任意值时，(input-1)*2 +4 -2*1 = 2*input，正好实现2倍上采样
#         self.transposed_conv = nn.Sequential(
#             nn.ConvTranspose2d(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 kernel_size=kernel_size,
#                 stride=stride,
#                 padding=padding,
#                 bias=False
#             ),
#             nn.BatchNorm2d(out_channels),
#             activation
#         )
        
#         # 融合卷积（与原结构保持一致）
#         self.fusion_conv = nn.Sequential(
#             nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             activation,
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             activation,
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
#         )

#     def forward(
#             self, x: torch.Tensor, skip: Optional[torch.Tensor] = None
#     ) -> torch.Tensor:
#         # 反卷积：同时完成上采样（2倍）和通道匹配
#         x = self.transposed_conv(x)

#         # 拼接跳跃连接特征（与原逻辑一致）
#         if skip is not None:
#             x = torch.cat([x, skip], dim=1)

#         # 融合特征（与原逻辑一致）
#         x = self.fusion_conv(x)
#         return x


# class CenterBlock(nn.Sequential):
#     """常规U-Net++中心块，处理编码器最深层特征"""

#     def __init__(
#             self,
#             in_channels: int,
#             out_channels: int,
#             activation: nn.Module = nn.ReLU(inplace=True),
#             dropout: float = 0.1,
#     ):
#         super().__init__(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             activation,
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             activation,
#             nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
#         )


# class UNetPlusPlusDecoder(nn.Module):
#     """常规U-Net++解码器"""

#     def __init__(
#             self,
#             encoder_channels: Sequence[int],
#             decoder_channels: Sequence[int],
#             activation: nn.Module = nn.ReLU(inplace=True),
#             dropout: float = 0.1,
#             interpolation_mode: str = "bilinear",
#             center: bool = False,
#     ):
#         super().__init__()

#         # 校验输入参数
#         if len(decoder_channels) != len(encoder_channels) - 1:
#             raise ValueError(
#                 f"Decoder channels length ({len(decoder_channels)}) must be len(encoder_channels)-1 ({len(encoder_channels) - 1})"
#             )

#         # 处理编码器通道（移除第一个特征，反转顺序）
#         encoder_channels = encoder_channels[1:]  # 移除第一个特征
#         encoder_channels = encoder_channels[::-1]  # 反转顺序，从最深层到浅层

#         # 计算解码器通道参数
#         head_channels = encoder_channels[0]
#         self.in_channels = [head_channels] + list(decoder_channels[:-1])
#         self.skip_channels = list(encoder_channels[1:]) + [0]
#         self.out_channels = decoder_channels
#         self.depth = len(self.in_channels) - 1

#         # 中心块
#         if center:
#             self.center = CenterBlock(
#                 head_channels,
#                 head_channels,
#                 activation=activation,
#                 dropout=dropout
#             )
#         else:
#             self.center = nn.Identity()

#         # 构建解码器块参数
#         self.kwargs = dict(
#             activation=activation,
#             dropout=dropout,
#             # interpolation_mode=interpolation_mode
#         )

#         # 构建密集连接块 - 保留U-Net++的核心特征
#         self.blocks = nn.ModuleDict()
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(layer_idx + 1):
#                 if depth_idx == 0:
#                     in_ch = self.in_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1)
#                     out_ch = self.out_channels[layer_idx]
#                 else:
#                     out_ch = self.skip_channels[layer_idx]
#                     skip_ch = self.skip_channels[layer_idx] * (layer_idx + 1 - depth_idx)
#                     in_ch = self.skip_channels[layer_idx - 1]

#                 self.blocks[f"x_{depth_idx}_{layer_idx}"] = DecoderBlock(
#                     in_ch, skip_ch, out_ch, **self.kwargs
#                 )

#         # 最后一个块
#         self.blocks[f"x_{0}_{len(self.in_channels) - 1}"] = DecoderBlock(
#             self.in_channels[-1], 0, self.out_channels[-1],** self.kwargs
#         )

#     def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
#         # 处理编码器输出特征
#         features = features[1:]  # 移除第一个特征
#         features = features[::-1]  # 反转顺序

#         # 中心块处理
#         x = self.center(features[0])
#         features[0] = x

#         # 密集连接特征融合 - 保留U-Net++的核心逻辑
#         dense_x = {}
#         for layer_idx in range(len(self.in_channels) - 1):
#             for depth_idx in range(self.depth - layer_idx):
#                 if layer_idx == 0:
#                     # 第一层直接处理编码器特征
#                     output = self.blocks[f"x_{depth_idx}_{depth_idx}"](
#                         features[depth_idx], features[depth_idx + 1]
#                     )
#                     dense_x[f"x_{depth_idx}_{depth_idx}"] = output
#                 else:
#                     # 深层处理：融合历史特征
#                     dense_l_i = depth_idx + layer_idx
#                     cat_features = [
#                         dense_x[f"x_{idx}_{dense_l_i}"]
#                         for idx in range(depth_idx + 1, dense_l_i + 1)
#                     ]
#                     cat_features = torch.cat(
#                         cat_features + [features[dense_l_i + 1]], dim=1
#                     )
#                     dense_x[f"x_{depth_idx}_{dense_l_i}"] = self.blocks[
#                         f"x_{depth_idx}_{dense_l_i}"
#                     ](dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)

#         # 最终输出
#         dense_x[f"x_{0}_{self.depth}"] = self.blocks[f"x_{0}_{self.depth}"](
#             dense_x[f"x_{0}_{self.depth - 1}"]
#         )
#         return dense_x[f"x_{0}_{self.depth}"]


# --------------------------SegHead--------------------------
class ConvNeXtSegHead(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            upsampling: int = 2,
            activation: nn.Module = None
    ):
        super().__init__()

        # 1. 处理激活函数：默认用GELU（ConvNeXt标准），用户可自定义或传None禁用
        self.activation = activation if activation is not None else nn.GELU()
        # 若完全不需要激活，用户可传 nn.Identity()

        # 2. 卷积层：加Padding确保分辨率不变（避免尺寸缩小）
        self.projection1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.projection2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        # 3. 上采样层：分辨率放大
        self.upsample_layer_1 = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )

        self.upsample_layer_2 = (
            nn.UpsamplingBilinear2d(scale_factor=upsampling)
            if upsampling > 1
            else nn.Identity()
        )

    def forward(self, x):
        x = self.projection1(x)  # 通道转换
        x = self.upsample_layer_1(x)  # 上采样
        x = self.projection2(x)  # 通道转换
        x = self.upsample_layer_2(x)  # 上采样

        return x


# ---------------------------------------------------
class ConvNeXtUNetPlusPlus(nn.Module):
    """SMP 风格的 ConvNeXt-Base-1k + U-Net++ 完整模型"""

    def __init__(self, in_channels=3, classes=7, pretrained_encoder=True,
                 decoder_channels=[768,384,192], decoder_dropout=0.1,
                 seg_head_activation=None):
        """
        Args:
            in_channels: 输入图像通道数（默认 3，RGB）
            classes: 分割类别数（默认 1，二分类）
            pretrained_encoder: 是否加载 Encoder 预训练权重
            decoder_channels: Decoder 各阶段通道数
            decoder_dropout: Decoder  dropout 概率
            seg_head_activation: SegHead 最终激活函数（如 nn.Sigmoid()）
        """
        super().__init__()
        # 1. 初始化 Encoder
        self.encoder = convnext.convnext_small(pretrained=pretrained_encoder)

        self.decoder = UnetPlusPlusDecoder(
            encoder_channels=self.encoder.out_channels,  # 编码器的通道配置（需与编码器一致）
            decoder_channels=decoder_channels,  # 解码器的输出通道配置
            # encoder_depth=3
            n_blocks=3
            # activation=nn.GELU(),  # 激活函数（默认GELU，可替换为nn.ReLU()等）
            # dropout=decoder_dropout,  # dropout概率（防止过拟合，0表示不使用）
            # interpolation_mode="bilinear",  # 上采样模式（bilinear/nearest，bilinear更平滑）
            # center=True  # 是否使用中心块（处理编码器最深层特征，建议开启）
        )
        self.segmentation_head = ConvNeXtSegHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=nn.GELU()
        )

    def forward(self, x):
        """完整前向流程：Encoder → Decoder → SegHead"""
        # 1. Encoder 提取特征
        features = self.encoder(x)  # [f0, f1, f2, f3]

        # 2. Decoder 融合特征
        decoder_out = self.decoder(features)  # (N, decoder_channels[-1], H/2, W/2)

        # 3. SegHead 输出分割结果
        seg_map = self.segmentation_head(decoder_out)  # (N, classes, H, W)

        return seg_map
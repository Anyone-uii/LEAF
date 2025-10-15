import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import convnextunetpluslplus

class AttentionFusion(nn.Module):
    """层级特征注意力融合模块：对两个同层级特征进行融合"""

    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力：学习通道维度的权重
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力：学习空间维度的权重
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),  # 输入两个特征的通道均值
            nn.Sigmoid()
        )

        # 融合后通道调整
        self.fusion_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, feat1, feat2):
        # 1. 通道注意力计算
        concat_feat = torch.cat([feat1, feat2], dim=1)  # (B, 2*C, H, W)
        channel_weight = self.channel_att(concat_feat)  # (B, C, 1, 1)

        # 2. 空间注意力计算（基于两个特征的均值）
        feat1_mean = torch.mean(feat1, dim=1, keepdim=True)  # (B, 1, H, W)
        feat2_mean = torch.mean(feat2, dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_weight = self.spatial_att(torch.cat([feat1_mean, feat2_mean], dim=1))  # (B, 1, H, W)

        # 3. 应用注意力权重
        feat1_weighted = feat1 * channel_weight * spatial_weight
        feat2_weighted = feat2 * channel_weight * (1 - spatial_weight)  # 互补空间权重

        # 4. 融合特征并调整通道数
        fused = torch.cat([feat1_weighted, feat2_weighted], dim=1)
        fused = self.fusion_conv(fused)  # 从2C降维到C，与原特征通道一致

        return fused

class DualUNet(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet",
                 in_channels=3, classes=9, hyper_c=0.2,feature_dim=512):
        super().__init__()
        # 编码器1和编码器2
        # self.model1 = convnextunetpluslplus.ConvNeXtUNetPlusPlus(
        #     in_channels=3,  # 输入通道数（RGB）
        #     classes=classes,  # 类别数（二分类）
        #     pretrained_encoder=True,  # 加载 Encoder 预训练权重
        #     decoder_channels=[512,256,128],  # Decoder 各阶段通道数
        #     decoder_dropout=0.1,  # Decoder dropout 概率
        #     # seg_head_activation=nn.Sigmoid()  # 二分类用 Sigmoid 激活
        # )

        # self.model2 = convnextunetpluslplus.ConvNeXtUNetPlusPlus(
        #     in_channels=3,  # 输入通道数（RGB）
        #     classes=classes,  # 类别数（二分类）
        #     pretrained_encoder=True,  # 加载 Encoder 预训练权重
        #     decoder_channels=[512,256,128],  # Decoder 各阶段通道数
        #     decoder_dropout=0.1,  # Decoder dropout 概率
        #     # seg_head_activation=nn.Sigmoid()  # 二分类用 Sigmoid 激活
        # )

        self.model1 = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
        self.model2 = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )

        # 获取编码器
        self.encoder1 = self.model1.encoder
        self.encoder2 = self.model2.encoder

        # 解码器和分割头复用原模型的
        self.decoder = self.model1.decoder
        self.segmentation_head = self.model1.segmentation_head


        # 初始化每级特征的注意力融合模块
        encoder_channels = self.encoder1.out_channels
        self.fusion_blocks = nn.ModuleList([
            AttentionFusion(enc_ch) for enc_ch in encoder_channels
        ])

    def forward(self, x1, x2):
        # 1. 两个编码器分别提取特征
        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)

        # 2. 每级特征注意力融合
        euclidean_fused_features = []
        for i in range(len(features1)):
            fused_feat = self.fusion_blocks[i](features1[i], features2[i])
            euclidean_fused_features.append(fused_feat)

        # 5. 融合后的特征传入解码器
        decoder_output = self.decoder(euclidean_fused_features)

        # 6. 分割头输出结果
        mask = self.segmentation_head(decoder_output)

        # 推理模式下只返回掩码和双曲特征
        return mask, features1, features2, euclidean_fused_features

# 测试模型
if __name__ == "__main__":
    # 创建模型
    model = DualUNet(classes=9)

    # 创建测试输入和标签
    x1 = torch.randn(2, 3, 512,512)
    x2 = torch.randn(2, 3, 512,512)
    labels = torch.randint(0, 9, (2, 512,512))  # 随机标签

    # 训练模式下前向传播
    output = model(x1, x2)

    # 推理模式下前向传播
    with torch.no_grad():
        output = model(x1, x2)
        print(f": {output}")

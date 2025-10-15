import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import segmentation_models_pytorch as smp
import math

# ---------------------- 原模型核心类定义（必须与训练时一致）----------------------

class FeatureCompressor(nn.Module):
    """将特征压缩到(B, D)维度，范数约束在[0,1]且保留差异"""

    def __init__(self, in_channels, out_dim=512, eps=1e-8):
        super().__init__()
        self.out_dim = out_dim
        self.eps = eps  # 避免除零

        # 1. 全局池化压缩空间维度
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 2. 特征投影与归一化链（稳定数值但不强制范数）
        self.projection = nn.Sequential(
            nn.Linear(in_channels, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        # x shape: (B, C, H, W)
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, C)

        # 投影得到高维特征
        x = self.projection(x)  # (B, out_dim)

        return x
    
# 双曲空间映射与距离计算工具
class HyperbolicUtils(nn.Module):
    """实现欧式空间与双曲空间的映射及距离计算"""

    def __init__(self, c=1.0):
        super().__init__()
        self.c = c  # 双曲空间曲率参数
        self.eps = 1e-8  # 数值稳定性参数
        self.sqrt_c = math.sqrt(c)

    def euclid_to_hyper(self, x):
        """将欧式向量映射到双曲空间，确保范数不溢出"""
        norm_x = torch.norm(x, dim=1, keepdim=True, p=2)  # 计算欧式范数 (B, 1)
        max_allowed_norm = 1.0 / math.sqrt(self.c) - self.eps  # 双曲空间允许的最大范数
        
        # 缩放因子：若范数超过阈值，则按比例缩小；否则保持不变
        scale = torch.min(torch.ones_like(norm_x), max_allowed_norm / (norm_x + self.eps))
        x_scaled = x * scale  # 缩放后范数 ≤ max_allowed_norm
        
        # 原有双曲映射公式（此时x_scaled的范数已安全）
        scaled_norm = math.sqrt(self.c) * torch.norm(x_scaled, dim=1, keepdim=True, p=2)
        tanh_term = torch.tanh(scaled_norm)
        safe_norm = torch.norm(x_scaled, dim=1, keepdim=True, p=2) + self.eps
        return (tanh_term / (math.sqrt(self.c) * safe_norm)) * x_scaled

    def hyper_to_euclid(self, x):
        """将双曲空间向量映射回欧式空间（对数映射）"""
        norm_x = torch.norm(x, dim=1, keepdim=True, p=2).clamp(min=self.eps, max=1 - self.eps)
        return (1 / math.sqrt(self.c)) * torch.arctanh(math.sqrt(self.c) * norm_x) * (x / norm_x)

    def hyper_distance(self, u, v):
        """计算双曲空间中两点u和v之间的距离（修正范数溢出）"""
        max_norm = (1.0 / math.sqrt(self.c)) - self.eps  # 动态根据c设置最大范数
        norm_u = torch.norm(u, dim=1, keepdim=True, p=2).clamp(max=max_norm)  # 钳位到圆盘内
        norm_v = torch.norm(v, dim=1, keepdim=True, p=2).clamp(max=max_norm)
        
        uv = torch.sum(u * v, dim=1, keepdim=True)  # 点积
        
        # 庞加莱圆盘距离公式（确保分母为正）
        numerator = 1 + 2 * self.c * torch.sum((u - v) **2, dim=1, keepdim=True)
        denominator = (1 - self.c * norm_u** 2) * (1 - self.c * norm_v **2)
        # 确保acosh输入≥1（公式数学性质保证，但加钳位防数值误差）
        arg = (numerator / denominator).clamp(min=1 + self.eps)
        distance = (1 / math.sqrt(self.c)) * torch.acosh(arg)
        
        return distance.mean()

    def hyper_distance_scale(self, u, v):
        """计算双曲空间中两点u和v之间的距离（先保角缩放向量，避免范数溢出）"""
        max_norm = (1.0 / math.sqrt(self.c)) - self.eps  # 双曲圆盘合法范数上限（保安全余量）
        eps = self.eps  # 避免除零
        
        # --------------------------
        # 第一步：对u和v进行保角范数缩放（保持夹角，约束范数≤max_norm）
        # --------------------------
        # 缩放u：保角，范数≤max_norm
        norm_u_original = torch.norm(u, dim=1, keepdim=True, p=2) + eps  # 原始范数（加eps防除零）
        target_norm_u = torch.min(norm_u_original, torch.full_like(norm_u_original, max_norm))  # 目标范数（不超上限）
        scale_u = target_norm_u / norm_u_original  # 保角缩放因子（k>0，方向不变）
        u_scaled = u * scale_u  # 缩放后的u（范数合规，夹角不变）
        
        # 缩放v：保角，范数≤max_norm
        norm_v_original = torch.norm(v, dim=1, keepdim=True, p=2) + eps
        target_norm_v = torch.min(norm_v_original, torch.full_like(norm_v_original, max_norm))
        scale_v = target_norm_v / norm_v_original
        v_scaled = v * scale_v  # 缩放后的v（范数合规，夹角不变）
        
        # --------------------------
        # 第二步：用缩放后的向量计算双曲距离
        # --------------------------
        # 计算缩放后的范数（已≤max_norm，可省略clamp，保留仅防数值误差）
        norm_u = torch.norm(u_scaled, dim=1, keepdim=True, p=2).clamp(max=max_norm)
        norm_v = torch.norm(v_scaled, dim=1, keepdim=True, p=2).clamp(max=max_norm)
        
        # 计算缩放后的点积（因保角，夹角不变，点积比例与原始一致）
        uv = torch.sum(u_scaled * v_scaled, dim=1, keepdim=True)
        
        # 庞加莱圆盘距离公式（基于缩放后的向量，确保分母为正）
        numerator = 1 + 2 * self.c * torch.sum((u_scaled - v_scaled) **2, dim=1, keepdim=True)
        denominator = (1 - self.c * norm_u** 2) * (1 - self.c * norm_v **2)
        
        # 确保acosh输入≥1（防数值误差，公式本身保证≥1）
        arg = (numerator / denominator).clamp(min=1 + eps)
        distance = (1 / math.sqrt(self.c)) * torch.acosh(arg)
        
        return distance.mean()

    def distance_to_origin(self, x):
        """计算双曲空间中点x到原点的距离（庞加莱圆盘模型）"""
        norm_x = torch.norm(x, dim=1, keepdim=True, p=2)  # (B, 1)，每个样本的欧氏范数
        # 钳位上限为 1/sqrt(c) - eps，确保点在圆盘内
        max_norm = (1.0 / math.sqrt(self.c)) - self.eps
        # norm_x_clamped = norm_x.clamp(max=max_norm)  # 避免超出双曲空间定义域
        scale = torch.min(torch.ones_like(norm_x), max_norm / (norm_x + self.eps))
        norm_x_scaled = norm_x * scale  # 放缩后的范数，严格 ≤ max_norm
        
        # 计算距离（保留批次维度，不提前取平均）
        distance = (1.0 / math.sqrt(self.c)) * torch.arctanh(math.sqrt(self.c) * norm_x_scaled)
        return distance.mean()  # 形状 (B, 1)，每个样本的距离
    
    def distance_to_origin_norm(self, x):
        """计算双曲空间中点x到原点的距离（庞加莱圆盘模型）"""
        norm_x = torch.norm(x, dim=1, keepdim=True, p=2)  # (B, 1)，每个样本的欧氏范数
        return norm_x.mean()  # 形状 (B, 1)，每个样本的距离

    def mobius_add(self, u, v):
        """莫比乌斯加法：双曲空间中的加法运算"""
        # 确保输入在双曲空间内
        u = u / (1 + 1e-8)
        v = v / (1 + 1e-8)

        norm_u_sq = torch.sum(u ** 2, dim=1, keepdim=True)
        norm_v_sq = torch.sum(v ** 2, dim=1, keepdim=True)
        uv_dot = torch.sum(u * v, dim=1, keepdim=True)

        # 莫比乌斯加法公式
        numerator = (1 + 2 * self.sqrt_c * uv_dot + self.c * norm_v_sq) * u + \
                    (1 - self.c * norm_u_sq) * v
        denominator = 1 + 2 * self.sqrt_c * uv_dot + (self.c ** 2) * norm_u_sq * norm_v_sq

        return numerator / (denominator + 1e-8)

class ShallowAttentionFusion(nn.Module):
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

class DeepAttentionFusion(nn.Module):
    """
    双模态特征融合模块（基于注意力机制）
    输入：两个特征图 x (B, C, H, W) 和 y (B, C, H, W)
    输出：融合后的特征图 (B, C, H, W)
    """
    def __init__(self, 
                 hidden_size,  # 特征通道数（输入x和y的通道数需与此一致）
                 num_heads=8,  # 注意力头数
                 mlp_dim=2048,  # MLP中间层维度
                 dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size

        # 1. 注意力层（双模态独立的Q/K/V）
        self.query_x = nn.Linear(hidden_size, self.all_head_size)
        self.key_x = nn.Linear(hidden_size, self.all_head_size)
        self.value_x = nn.Linear(hidden_size, self.all_head_size)
        
        self.query_y = nn.Linear(hidden_size, self.all_head_size)
        self.key_y = nn.Linear(hidden_size, self.all_head_size)
        self.value_y = nn.Linear(hidden_size, self.all_head_size)
        
        self.out_x = nn.Linear(hidden_size, hidden_size)
        self.out_y = nn.Linear(hidden_size, hidden_size)

        # 2. 动态融合权重（可学习参数）
        self.w11 = nn.Parameter(torch.tensor(0.5))  # x自注意力权重
        self.w12 = nn.Parameter(torch.tensor(0.5))  # x跨模态注意力权重
        self.w21 = nn.Parameter(torch.tensor(0.5))  # y自注意力权重
        self.w22 = nn.Parameter(torch.tensor(0.5))  # y跨模态注意力权重

        # 3. 层归一化与dropout
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)
        self.norm_x1 = nn.LayerNorm(hidden_size)  # 注意力前归一化
        self.norm_y1 = nn.LayerNorm(hidden_size)
        self.norm_x2 = nn.LayerNorm(hidden_size)  # MLP前归一化
        self.norm_y2 = nn.LayerNorm(hidden_size)

        # 4. MLP层（增强特征非线性表达）
        self.mlp_x = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )
        self.mlp_y = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout_rate)
        )

    def transpose_for_scores(self, x):
        """将特征拆分到多个注意力头"""
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # (B, heads, N, head_dim)

    def forward(self, x, y):
        # 输入特征图形状：(B, C, H, W) → 转换为序列：(B, N, C)，其中N=H*W
        B, C, H, W = x.shape
        N = H * W
        x_seq = x.flatten(2).transpose(1, 2)  # (B, N, C)
        y_seq = y.flatten(2).transpose(1, 2)  # (B, N, C)

        # 残差连接备份
        residual_x = x_seq
        residual_y = y_seq

        # -------------------------- 注意力融合阶段 --------------------------
        # 层归一化
        x_norm = self.norm_x1(x_seq)
        y_norm = self.norm_y1(y_seq)

        # 生成Q/K/V
        qx = self.query_x(x_norm)
        kx = self.key_x(x_norm)
        vx = self.value_x(x_norm)
        
        qy = self.query_y(y_norm)
        ky = self.key_y(y_norm)
        vy = self.value_y(y_norm)

        # 拆分注意力头
        qx = self.transpose_for_scores(qx)  # (B, heads, N, head_dim)
        kx = self.transpose_for_scores(kx)
        vx = self.transpose_for_scores(vx)
        
        qy = self.transpose_for_scores(qy)
        ky = self.transpose_for_scores(ky)
        vy = self.transpose_for_scores(vy)

        # 1. 自注意力
        # RGB自注意力
        attn_scores_x = torch.matmul(qx, kx.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_x = F.softmax(attn_scores_x, dim=-1)
        attn_probs_x = self.attn_dropout(attn_probs_x)
        attn_output_x = torch.matmul(attn_probs_x, vx)  # (B, heads, N, head_dim)
        
        # Depth自注意力
        attn_scores_y = torch.matmul(qy, ky.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_y = F.softmax(attn_scores_y, dim=-1)
        attn_probs_y = self.attn_dropout(attn_probs_y)
        attn_output_y = torch.matmul(attn_probs_y, vy)  # (B, heads, N, head_dim)

        # 2. 跨模态注意力
        # RGB查询→Depth键值
        attn_scores_cx = torch.matmul(qx, ky.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_cx = F.softmax(attn_scores_cx, dim=-1)
        attn_probs_cx = self.attn_dropout(attn_probs_cx)
        attn_output_cx = torch.matmul(attn_probs_cx, vy)  # (B, heads, N, head_dim)
        
        # Depth查询→RGB键值
        attn_scores_cy = torch.matmul(qy, kx.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_cy = F.softmax(attn_scores_cy, dim=-1)
        attn_probs_cy = self.attn_dropout(attn_probs_cy)
        attn_output_cy = torch.matmul(attn_probs_cy, vx)  # (B, heads, N, head_dim)

        # 3. 拼接注意力头并投影
        # 自注意力结果
        attn_output_x = attn_output_x.permute(0, 2, 1, 3).contiguous()
        attn_output_x = attn_output_x.view(B, N, self.all_head_size)
        attn_output_x = self.out_x(attn_output_x)
        attn_output_x = self.proj_dropout(attn_output_x)
        
        attn_output_y = attn_output_y.permute(0, 2, 1, 3).contiguous()
        attn_output_y = attn_output_y.view(B, N, self.all_head_size)
        attn_output_y = self.out_y(attn_output_y)
        attn_output_y = self.proj_dropout(attn_output_y)
        
        # 跨模态注意力结果
        attn_output_cx = attn_output_cx.permute(0, 2, 1, 3).contiguous()
        attn_output_cx = attn_output_cx.view(B, N, self.all_head_size)
        attn_output_cx = self.out_x(attn_output_cx)
        attn_output_cx = self.proj_dropout(attn_output_cx)
        
        attn_output_cy = attn_output_cy.permute(0, 2, 1, 3).contiguous()
        attn_output_cy = attn_output_cy.view(B, N, self.all_head_size)
        attn_output_cy = self.out_y(attn_output_cy)
        attn_output_cy = self.proj_dropout(attn_output_cy)

        # 4. 动态权重融合（自注意力+跨模态注意力）
        x_attn = self.w11 * attn_output_x + self.w12 * attn_output_cx
        y_attn = self.w21 * attn_output_y + self.w22 * attn_output_cy

        # 残差连接
        x_attn = x_attn + residual_x
        y_attn = y_attn + residual_y

        # -------------------------- MLP增强阶段 --------------------------
        # 层归一化
        x_mlp = self.norm_x2(x_attn)
        y_mlp = self.norm_y2(y_attn)

        # MLP处理
        x_mlp = self.mlp_x(x_mlp)
        y_mlp = self.mlp_y(y_mlp)

        # 残差连接
        x_final = x_mlp + x_attn
        y_final = y_mlp + y_attn

        # -------------------------- 特征融合输出 --------------------------
        # 双模态特征相加融合
        fused_seq = x_final + y_final  # (B, N, C)
        
        # 序列转换回特征图形状 (B, C, H, W)
        fused_feat = fused_seq.transpose(1, 2).view(B, C, H, W)
        
        return fused_feat

class HyperbolicAttentionFusion(nn.Module):
    """高效的双曲空间注意力融合模块（完全向量化实现）"""

    def __init__(self, in_channels, hyper_c=0.2, hidden_dim=512):
        super().__init__()
        self.hyper_utils = HyperbolicUtils(c=hyper_c)
        self.in_channels = in_channels

        # 用于将欧式特征映射到双曲空间的投影层
        self.to_hyperbolic1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
        self.to_hyperbolic2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        self.to_hyperbolic3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )

        # 距离激活函数（使用Sigmoid将距离转换为权重）
        self.distance_activation = nn.Sigmoid()

        self.distance_mlp = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=1),  # 空间感知的局部距离编码
        )

        self.ff = nn.Sequential(
            nn.Conv2d(2 * in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
        )

        self.tangent_norm = nn.LayerNorm([in_channels])  # 仅指定通道维度（H、W动态适配）

    def hyper_distance_map(self, u, v):
        """
        向量化计算双曲空间中两点u和v之间的距离图（修正范数溢出，保角缩放）
        参数:
            u: (B, C, H, W) 查询点特征
            v: (B, C, H, W) 目标特征
        返回:
            distance_map: (B, 1, H, W) 距离图
        """
        B, C, H, W = u.shape
        eps = 1e-6  # 数值稳定性参数
        c = self.hyper_utils.c  # 双曲空间曲率
        max_norm = (1.0 / math.sqrt(c)) - eps  # 动态合法范数上限（基于曲率c）

        # --------------------------
        # 第一步：重塑特征并进行保角范数缩放（约束范数≤max_norm，保持夹角）
        # --------------------------
        # 重塑为 (B, H, W, C)，方便按空间位置计算范数和缩放
        u_reshaped = u.permute(0, 2, 3, 1)  # (B, H, W, C)
        v_reshaped = v.permute(0, 2, 3, 1)  # (B, H, W, C)

        # 计算u的原始范数（每个空间位置的向量范数）
        norm_u_original = torch.norm(u_reshaped, dim=3, keepdim=True) + eps  # (B, H, W, 1)，加eps防除零
        # 目标范数：不超过max_norm（保角缩放，保留原始比例）
        target_norm_u = torch.min(norm_u_original, torch.full_like(norm_u_original, max_norm))
        # 保角缩放因子（k>0，方向不变）
        scale_u = target_norm_u / norm_u_original
        # 缩放后的u（范数合规，夹角不变）
        u_scaled = u_reshaped * scale_u  # (B, H, W, C)

        # 对v执行相同的保角缩放
        norm_v_original = torch.norm(v_reshaped, dim=3, keepdim=True) + eps  # (B, H, W, 1)
        target_norm_v = torch.min(norm_v_original, torch.full_like(norm_v_original, max_norm))
        scale_v = target_norm_v / norm_v_original
        v_scaled = v_reshaped * scale_v  # (B, H, W, C)

        # --------------------------
        # 第二步：基于缩放后的特征计算双曲距离图
        # --------------------------
        # 计算缩放后的范数（已≤max_norm，加钳位防浮点误差）
        norm_u = torch.norm(u_scaled, dim=3, keepdim=True).clamp(max=max_norm)  # (B, H, W, 1)
        norm_v = torch.norm(v_scaled, dim=3, keepdim=True).clamp(max=max_norm)  # (B, H, W, 1)

        # 计算缩放后的点积（保角性确保夹角不变，点积比例与原始一致）
        uv_dot = torch.sum(u_scaled * v_scaled, dim=3, keepdim=True)  # (B, H, W, 1)

        # 计算差向量的平方和（基于缩放后的特征）
        diff_norm_sq = torch.sum((u_scaled - v_scaled) **2, dim=3, keepdim=True)  # (B, H, W, 1)

        # 庞加莱圆盘距离公式（完全向量化）
        numerator = 1 + 2 * c * diff_norm_sq
        denominator = (1 - c * norm_u** 2) * (1 - c * norm_v **2)

        # 确保acosh输入≥1（防数值误差，公式数学性质保证≥1）
        inside_acosh = (numerator / denominator).clamp(min=1 + eps)

        # 计算距离
        distance = (1 / math.sqrt(c)) * torch.acosh(inside_acosh)  # (B, H, W, 1)

        # 重塑为 (B, 1, H, W) 距离图
        distance_map = distance.permute(0, 3, 1, 2)  # (B, 1, H, W)

        return distance_map

    def forward(self, euclidean_fused, feat1, feat2):
        """
        参数: 该模块可以被称为“”
            euclidean_fused: 欧式空间融合后的特征 (B, C, H, W)
            feat1: 模态1特征 (B, C, H, W)
            feat2: 模态2特征 (B, C, H, W)
        """
        B, C, H, W = euclidean_fused.shape

        # 1. 将欧式空间融合特征映射到双曲空间作为查询点
        query_hyper = self.hyper_utils.euclid_to_hyper(
            self.to_hyperbolic1(euclidean_fused).permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        ).reshape(euclidean_fused.size(0), euclidean_fused.size(2),
                  euclidean_fused.size(3), -1).permute(0, 3, 1, 2)

        # 2. 将两个模态特征也映射到双曲空间
        feat1_hyper = self.hyper_utils.euclid_to_hyper(
            self.to_hyperbolic2(feat1).permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        ).reshape(feat1.size(0), feat1.size(2), feat1.size(3), -1).permute(0, 3, 1, 2)

        feat2_hyper = self.hyper_utils.euclid_to_hyper(
            self.to_hyperbolic3(feat2).permute(0, 2, 3, 1).reshape(-1, self.in_channels)
        ).reshape(feat2.size(0), feat2.size(2), feat2.size(3), -1).permute(0, 3, 1, 2)

        # 3. 向量化计算查询点与两个模态特征的测地距离
        dist1_map = self.hyper_distance_map(query_hyper, feat1_hyper)  # (B, 1, H, W)
        dist2_map = self.hyper_distance_map(query_hyper, feat2_hyper)  # (B, 1, H, W)

        dist_sum = dist1_map + dist2_map + 1e-8  # 避免除零
        rel_dist1 = dist1_map / dist_sum  # 相对距离比例（越大表示该模态相对越远）
        rel_dist2 = dist2_map / dist_sum

        # 3. 使用softmax进行权重归一化（距离越小，权重越大）
        weights = torch.cat([-rel_dist1, -rel_dist2], dim=1)  # (B, 2, H, W)
        weights = self.distance_mlp(weights)
        normalized_weights = torch.softmax(weights, dim=1)
        weight1, weight2 = normalized_weights[:, 0:1, ...], normalized_weights[:, 1:2, ...]

        feat1_hyper_flat = feat1_hyper.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        feat2_hyper_flat = feat2_hyper.permute(0, 2, 3, 1).reshape(-1, C)  # (B*H*W, C)
        
        # 双曲点 → 切空间向量（线性空间，可自由加权）
        feat1_tangent = self.hyper_utils.hyper_to_euclid(feat1_hyper_flat)  # (B*H*W, C)
        feat2_tangent = self.hyper_utils.hyper_to_euclid(feat2_hyper_flat)  # (B*H*W, C)

        # 3.2 门控权重展平（匹配切空间向量形状）
        weight1_flat = weight1.permute(0, 2, 3, 1).reshape(-1, 1)  # (B*H*W, 1)
        weight2_flat = weight2.permute(0, 2, 3, 1).reshape(-1, 1)  # (B*H*W, 1)

        # 3.3 切空间欧式加权（权重无几何意义，直接线性组合）
        fused_tangent = weight1_flat * feat1_tangent + weight2_flat * feat2_tangent  # (B*H*W, C)

        # 4. 双曲融合特征→欧式空间（用于后续与原欧式融合特征拼接）
        fused_tangent = fused_tangent.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        # 直接使用self.tangent_norm，无需动态创建
        fused_euclidean = self.tangent_norm(fused_tangent.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # 5. 最终融合：双曲→欧式特征 + 原欧式融合特征
        fused_feature = torch.cat([fused_euclidean, euclidean_fused], dim=1)  # (B, 2C, H, W)
        fused_feature = self.ff(fused_feature)  # 压缩回C维

        return fused_feature



        # # 4. 在双曲空间中使用莫比乌斯加法进行加权融合
        # # 重塑特征以便应用莫比乌斯加法 (B*H*W, C, 1)
        # feat1_flat = feat1_hyper.permute(0, 2, 3, 1).reshape(-1, C, 1)  # (B*H*W, C, 1)
        # feat2_flat = feat2_hyper.permute(0, 2, 3, 1).reshape(-1, C, 1)  # (B*H*W, C, 1)

        # # 重塑权重 (B*H*W, 1, 1)
        # w1_flat = weight1.permute(0, 2, 3, 1).reshape(-1, 1, 1)  # (B*H*W, 1, 1)
        # w2_flat = weight2.permute(0, 2, 3, 1).reshape(-1, 1, 1)  # (B*H*W, 1, 1)

        # # 应用权重并进行莫比乌斯加法
        # weighted_feat1 = w1_flat * feat1_flat.transpose(1, 2)  # (B*H*W, 1, C)
        # weighted_feat2 = w2_flat * feat2_flat.transpose(1, 2)  # (B*H*W, 1, C)

        # # 莫比乌斯加法融合
        # fused_flat = self.hyper_utils.mobius_add(
        #     weighted_feat1.squeeze(1),  # (B*H*W, C)
        #     weighted_feat2.squeeze(1)  # (B*H*W, C)
        # ).unsqueeze(1)  # (B*H*W, 1, C)

        # # 恢复原始形状
        # fused_hyper = fused_flat.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        # # 5. 将融合后的双曲特征映射回欧式空间
        # fused_euclidean = self.hyper_utils.hyper_to_euclid(
        #     fused_hyper.permute(0, 2, 3, 1).reshape(-1, C)
        # ).reshape(B, H, W, C).permute(0, 3, 1, 2)

        # fused_feature = torch.cat([fused_euclidean, euclidean_fused], dim=1)
        # fused_feature = self.ff(fused_feature)

        # return fused_feature

def top1_loss(predicts, target_index):
    """
    Top-1 损失：让指定索引的元素成为最高分
    predicts: (B, L)
    target_index: (B,)
    """
    log_probs = torch.log_softmax(predicts, dim=1)  # (B, L)
    loss = -log_probs[torch.arange(predicts.size(0)), target_index]
    return loss.mean()

def listwise_ranking_loss(predicts, targets):
    """
    计算 ListMLE 损失。
    参数:
        predicts: 模型预测的得分 (B, L)，B是批次大小，L是列表长度（例如层次数量）。
                  得分越高，代表预测的排名应越靠前（即到原点距离越小）。
        targets: 真实的排序依据 (B, L)，例如到双曲空间原点的真实距离。
                  值越小，代表其真实的排名应越靠前。
    返回:
        losses: 标量，整个批次的平均 ListMLE 损失。
    """
    # 1. 根据真实标签（targets，即距离）进行降序排序。
    #    因为：距离越小（targets值越小）的样本，其真实排名越靠前。
    #    所以我们需要让 predicts 得分高的样本对应 targets 值小的样本。
    #    indices 的形状为 (B, L)，表示每个样本中，按真实距离从小到大排序后的索引。
    _, indices = torch.sort(targets, dim=1, descending=False) # 升序排序：最小的距离排第一（排名最高）

    # 2. 根据排序索引 indices，重排 predicts 的得分。
    #    这样 predicts_sorted 的第一列就是当前“应该”得分最高的样本，最后一列是“应该”得分最低的样本。
    predicts_sorted = torch.gather(predicts, dim=1, index=indices)

    # 3. 计算 ListMLE 损失的核心步骤
    #    计算累积和：从最后一个元素开始，逆向累积指数和。
    #    例如：对于序列 [s1, s2, s3]，计算过程为：
    #        cumsums = [log(exp(s3)+exp(s2)+exp(s1)), log(exp(s3)+exp(s2)), log(exp(s3))]
    #    这里使用 logsumexp 来数值稳定地计算 log(sum(exp(...)))。
    y = predicts_sorted
    # 反转张量，以便从排名最低的开始计算累积和
    y_reverse = y.flip(dims=[1])
    # 计算累积的 logsumexp
    cum_logsumexp = torch.logcumsumexp(y_reverse, dim=1)
    # 再次反转回来，使其与原始顺序对应
    cum_logsumexp = cum_logsumexp.flip(dims=[1])

    # 4. 计算每个位置的负对数似然损失： - (y_i - cum_logsumexp_i)
    #    注意：这里计算的是每个位置（排名）的损失
    loss_per_position = - (y - cum_logsumexp)
    # 对所有位置和所有批次求平均
    loss = loss_per_position.mean()

    return loss

class RankingBasedGeometryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, c=0.5):
        super().__init__()
        self.hyper_utils = HyperbolicUtils(c)
        self.alpha = alpha  # 层次深度约束权重
        self.beta = beta    # 融合特征核心度约束权重
        self.gamma = gamma  # 模态差异性约束权重

    def forward(self, hyper_fused, hyper_features1, hyper_features2):
        total_loss = 0.0
        num_levels = len(hyper_fused)

        # 1. 层次深度约束: 使用 ListMLE
        # 构建一个“理想”的层次排序：深层（索引大）的特征到原点距离应该更小，排名更高。
        # 我们假设一个“理想”的层次顺序：从深到浅（索引从大到小），其到原点的距离应该单调递增（即排名价值递减）。
        # 但我们没有绝对的“真实标签”，所以我们用层次索引的逆序作为“理想排名”的代理监督信号。
        # 例如，对于4个层次：[L0, L1, L2, L3]，我们期望 L3（最深）最接近原点，L0（最浅）最远。
        # 所以“理想”的排序顺序（从高到低）是 [L3, L2, L1, L0]。

        # 收集所有层次融合特征到原点的距离
        ideal_level_ranks = torch.arange(num_levels-1, -1, -1, device=hyper_fused[0].device)  # 形状 (num_levels,)
        # 模型预测得分：用特征到原点的距离的负值（距离越小，得分越高）
        predicted_level_scores = torch.stack([
            -self.hyper_utils.distance_to_origin_norm(feat) 
            for feat in hyper_fused
        ]) 

        # 增加批次维度以适配 ListMLE
        predicted_level_scores = predicted_level_scores.unsqueeze(0)  # (1, num_levels)
        ideal_level_ranks = ideal_level_ranks.unsqueeze(0)
        
        # 计算层次深度损失
        level_loss = listwise_ranking_loss(predicted_level_scores, ideal_level_ranks)
        total_loss += self.alpha * level_loss

        # --------------------------
        # 2. 融合特征核心度约束（修正部分）
        # --------------------------
        for i in range(num_levels):
            # 预设理想排序：融合特征（索引0）应排在前，单模态特征（索引1、2）排后
            # ideal_coreness_ranks = torch.tensor([0, 1, 1], device=hyper_fused[0].device)  # 形状 (3,)
            # 模型预测得分：距离的负值（距离越小，得分越高）
            predicted_core_scores = torch.stack([
                -self.hyper_utils.distance_to_origin_norm(hyper_fused[i]),
                -self.hyper_utils.distance_to_origin_norm(hyper_features1[i]),
                -self.hyper_utils.distance_to_origin_norm(hyper_features2[i])
            ]).unsqueeze(0)  # 形状 (3,)
            
            # 融合特征索引 = 0
            target_index = torch.tensor([0], device=predicted_core_scores.device).unsqueeze(0)
            coreness_loss = top1_loss(predicted_core_scores, target_index)
            total_loss += self.beta * coreness_loss
        
        return total_loss

class DualInputHyperbolicUNet(nn.Module):
    def __init__(self, encoder_name="resnet50", encoder_weights="imagenet",
                 in_channels=3, classes=7, hyper_c=0.1,feature_dim=512):
        super().__init__()
        
        # 编码器1和编码器2
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

        # self.model1 = convnextunetpluslplus.ConvNeXtUNetPlusPlus(
        #     in_channels=3,  # 输入通道数（RGB）
        #     classes=classes,  # 类别数（二分类）
        #     pretrained_encoder=False,  # 加载 Encoder 预训练权重
        #     decoder_channels=[768,384,192],  # Decoder 各阶段通道数
        #     decoder_dropout=0,  # Decoder dropout 概率
        #     # seg_head_activation=nn.Sigmoid()  # 二分类用 Sigmoid 激活
        # )

        # self.model2 = convnextunetpluslplus.ConvNeXtUNetPlusPlus(
        #     in_channels=3,  # 输入通道数（RGB）
        #     classes=classes,  # 类别数（二分类）
        #     pretrained_encoder=False,  # 加载 Encoder 预训练权重
        #     decoder_channels=[768,384,192],  # Decoder 各阶段通道数
        #     decoder_dropout=0,  # Decoder dropout 概率
        #     # seg_head_activation=nn.Sigmoid()  # 二分类用 Sigmoid 激活
        # )

        # 获取编码器
        self.encoder1 = self.model1.encoder
        self.encoder2 = self.model2.encoder

        # 解码器和分割头复用原模型的
        self.decoder = self.model1.decoder
        self.segmentation_head = self.model1.segmentation_head

        # 初始化每级特征的注意力融合模块
        encoder_channels = self.encoder1.out_channels
        self.fusion_blocks = nn.ModuleList([
            ShallowAttentionFusion(enc_ch) for enc_ch in encoder_channels
        ])

        self.mbafusion = DeepAttentionFusion(hidden_size =encoder_channels[-1])

        self.hyperbolic_fusion_blocks = nn.ModuleList([
            HyperbolicAttentionFusion(enc_ch, hyper_c) for enc_ch in encoder_channels
        ])

        # 添加特征压缩模块：将各层级特征统一到(B, feature_dim)
        self.feature_compressors1 = nn.ModuleList([
            FeatureCompressor(enc_ch, feature_dim) for enc_ch in encoder_channels
        ])
        self.feature_compressors2 = nn.ModuleList([
            FeatureCompressor(enc_ch, feature_dim) for enc_ch in encoder_channels
        ])
        self.feature_compressors_fused = nn.ModuleList([
            FeatureCompressor(enc_ch, feature_dim) for enc_ch in encoder_channels
        ])

        # 双曲空间工具
        self.hyper_utils = HyperbolicUtils(c=hyper_c)

        # 获取编码器层级数量
        self.num_levels = len(encoder_channels)

        # 几何损失函数
        self.geometry_loss_fn = RankingBasedGeometryLoss(c=hyper_c)

        # 全局信息注入
        self.compress_injectors = nn.ModuleList([
            # 每个注入器对应解码器某一层的尺寸，将feature_dim压缩到与解码器层通道数匹配
            nn.Sequential(
                nn.Linear(feature_dim, encoder_channels[i]),  # 匹配解码器层通道数
                nn.LayerNorm(encoder_channels[i])
            ) for i in range(self.num_levels)
        ])

    def forward(self, x1, x2):
        # 1. 两个编码器分别提取特征
        features1 = self.encoder1(x1)
        features2 = self.encoder2(x2)

        # 2. 每级特征注意力融合
        euclidean_fused_features = []
        for i in range(len(features1)-1):
            fused_feat = self.fusion_blocks[i](features1[i], features2[i])
            euclidean_fused_features.append(fused_feat)
        
        deepfused_feat = self.mbafusion(features1[-1], features2[-1])
        euclidean_fused_features.append(deepfused_feat)


        hyperbolic_fused_features = []
        for i in range(len(features1)):
            hyperbolic_fused = self.hyperbolic_fusion_blocks[i](
                euclidean_fused_features[i], features1[i], features2[i]
            )
            hyperbolic_fused_features.append(hyperbolic_fused)

        # 3. 将所有特征图压缩到相同的(B, D)尺寸
        # 单模态特征压缩
        compressed1 = [self.feature_compressors1[i](feat)
                       for i, feat in enumerate(features1)]
        compressed2 = [self.feature_compressors2[i](feat)
                       for i, feat in enumerate(features2)]

        # 融合特征压缩
        compressed_fused = [self.feature_compressors_fused[i](feat)
                            for i, feat in enumerate(hyperbolic_fused_features)]

        # 4.1 将压缩后的特征映射到双曲空间
        hyper_features1 = [self.hyper_utils.euclid_to_hyper(feat) for feat in compressed1]
        hyper_features2 = [self.hyper_utils.euclid_to_hyper(feat) for feat in compressed2]
        hyper_fused = [self.hyper_utils.euclid_to_hyper(feat) for feat in compressed_fused]

        # 4.2 将压缩向量注入到对应的解码器输入特征中
        decoder_inputs = []
        for i in range(self.num_levels):
            # 当前解码器输入特征图：hyperbolic_fused_features[i] → (B, C, H, W)
            feat_map = hyperbolic_fused_features[i]
            B, C, H, W = feat_map.shape

            # 压缩向量融合（取fused的压缩特征，或1+2的均值，增强全局信息）
            global_vec = (compressed1[i] + compressed2[i] + compressed_fused[i]) / 3  # (B, 512)
            # 向量→特征图：先线性映射到通道数C，再扩展空间维度（B, C）→ (B, C, 1, 1) → (B, C, H, W)
            global_feat = self.compress_injectors[i](global_vec).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            global_feat = global_feat.expand(-1, -1, H, W)  # (B, C, H, W)

            # 融合全局信息与局部特征图（元素相加或卷积融合）
            fused_with_global = feat_map + global_feat  # 简单高效，适合初期
            decoder_inputs.append(fused_with_global)


        # 5. 融合后的特征传入解码器
        decoder_output = self.decoder(decoder_inputs)

        # 6. 分割头输出结果
        mask = self.segmentation_head(decoder_output)

        geom_loss = self.geometry_loss_fn(hyper_fused, hyper_features1, hyper_features2)

        # 推理模式下只返回掩码和双曲特征
        return mask, geom_loss, hyper_features1 +  hyper_features2 + hyper_fused

# ---------------------- 核心功能函数（双图像输入适配）----------------------
def preprocess_two_images(img1_path, img2_path):
    """预处理两张图像，返回两个模型输入张量"""
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.PILToTensor(),  # 转换为Tensor
    ])

    # 处理第一张图
    img1 = Image.open(img1_path).convert("RGB")
    img1_tensor = transform(img1).unsqueeze(0)  # (1, 3, H, W)

    # 处理第二张图
    img2 = Image.open(img2_path).convert("RGB")
    img2_tensor = transform(img2).unsqueeze(0)  # (1, 3, H, W)

    return img1_tensor/255.0, img2_tensor/255.0, img1, img2  # 返回张量和原始图像


def preprocess_two_images_whu(img1_path, img2_path):
    """预处理两张图像，返回两个模型输入张量"""
    transform = transforms.Compose([
        transforms.Resize([256,256]),
        transforms.PILToTensor(),  # 转换为Tensor
    ])

    # 处理第一张图
    img1 = Image.open(img1_path).convert("RGB")
    img1_tensor = transform(img1).unsqueeze(0)  # (1, 3, H, W)

    # 处理第二张图
    img2 = Image.open(img2_path).convert("RGB")
    img2_tensor = transform(img2).unsqueeze(0)  # (1, 3, H, W)

    return img1_tensor/255.0, img2_tensor/255.0, img1, img2  # 返回张量和原始图像


def load_trained_model(model_path, encoder_name="resnet50", classes=9, hyper_c=0.1, feature_dim=512):
    """加载.pth.tar格式模型（适配你的保存方式：state_dict对应模型参数）"""
    # 初始化模型结构（与训练时一致）
    model = DualInputHyperbolicUNet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=classes,
        hyper_c=hyper_c,
        feature_dim=feature_dim
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载.pth.tar文件（你的保存格式：内部是包含'state_dict'的字典）
    checkpoint = torch.load(model_path, map_location=device)
    print(f".pth.tar文件内容键名：{checkpoint.keys()}")  # 确认包含'state_dict'等键
    
    # 直接提取模型状态字典（你的保存方式中，键是'state_dict'）
    model_state_dict = checkpoint['state_dict']
    print(f"已提取'state_dict'键对应的模型参数")
    
    # 加载参数到模型
    model.load_state_dict(model_state_dict)
    model.eval()  # 推理模式
    model.to(device)
    print(f".pth.tar模型加载完成，设备：{device}")
    return model, device


# ---------------------- 主函数（双图像输入流程）----------------------
#NH49E001017_3_13
#NH49E008024_7_12

def main():
    # 【用户需调整的参数】
    MODEL_PATH = "/home/hfz/doc/HyFusion/workdir/Hyfusion_model_WHU_res_c-1.pth.tar"   # 模型权重路径
    IMAGE1_PATH = "/home/hfz/data/WHU-OPT-SAR/optical256/NH49E008024_7_12.tif"             # 第一张图像路径（如RGB图）
    IMAGE2_PATH = "/home/hfz/data/WHU-OPT-SAR/sar256/NH49E008024_7_12.tif"             # 第二张图像路径（如Depth图）
    ENCODER_NAME = "resnet50"                # 与训练时一致
    CLASSES = 8                               # 与训练时一致
    HYPER_C = 0.1                            # 与训练时一致
    FEATURE_DIM = 512                         # 与训练时一致
    SAVE_DISK_PATH = "/home/hfz/doc/HyFusion/workdir/NH49E008024_7_12-5.png"  # 保存路径

    cmap_9class = [
        (0, 0, 0),          # 0: unlabelled（未标注）
        (64, 0, 128),       # 1: car（车辆）
        (64, 64, 0),        # 2: person（行人）
        (0, 128, 192),      # 3: bike（自行车）
        (0, 0, 192),        # 4: curve（弯道）
        (128, 128, 0),      # 5: car_stop（停车区）
        (64, 64, 128),      # 6: guardrail（护栏）
        (192, 128, 128),    # 7: color_cone（彩色锥桶）
        (192, 64, 0)        # 8: bump（减速带）
    ]

    cmap_8class = [
        (0, 0, 0),          # 0: unlabelled（未标注）
        (165, 82, 42),      # 1: farmland
        (220, 20, 60),      # 2: city
        (220, 20, 60),      # 3: village
        (0, 100, 255),      # 4: water
        (34, 139, 34),      # 5: forest
        (0, 255, 255),      # 6: road
        (128, 0, 128),      # 7: others
    ]


    # 1. 预处理两张图像
    print("1. 预处理两张图像...")
    x1_tensor, x2_tensor, img1, img2 = preprocess_two_images_whu(IMAGE1_PATH, IMAGE2_PATH)

    print(f"图像1预处理后形状：{x1_tensor.shape}")
    print(f"图像2预处理后形状：{x2_tensor.shape}")

    # 2. 加载模型
    print("\n2. 加载训练好的模型...")
    model, device = load_trained_model(
        model_path=MODEL_PATH,
        encoder_name=ENCODER_NAME,
        classes=CLASSES,
        hyper_c=HYPER_C,
        feature_dim=FEATURE_DIM
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    x1_tensor = x1_tensor.to(device) 
    x2_tensor = x2_tensor.to(device)

    # 3. 提取双曲特征
    print("\n3. 推理.......")
    mask, _, _ = model(x1_tensor, x2_tensor)
    # 1. 去掉batch维度（如果有），得到 (9, H, W)
    mask_squeezed = mask.squeeze(0)  # 假设batch_size=1，去掉第0维
    # 2. 在类别维度（dim=0）上求argmax，得到每个像素的类别，形状为 (H, W)
    mask = torch.argmax(mask_squeezed, dim=0)  # 关键：指定dim=0

    unique_values = torch.unique(mask)
    print(unique_values)

    mask_np = mask.cpu().numpy()
    cmap_np = np.array(cmap_8class, dtype=np.uint8)
    print(cmap_np)
    colored_mask = cmap_np[mask_np]

    # 3. 转换为PIL图像并保存
    # 注意：若cmap包含透明度通道（RGBA），需将Image模式改为"RGBA"
    img = Image.fromarray(colored_mask, mode="RGB")  # 若为RGBA则用"RGBA"
    img.save(SAVE_DISK_PATH)



if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import math
import model_3Dunet


class FeatureCompressor3D(nn.Module):
    """3D特征压缩模块：将3D特征压缩到(B, D)维度，范数约束在[0,1]且保留差异"""

    def __init__(self, in_channels, out_dim=64, eps=1e-8):
        super().__init__()
        self.out_dim = out_dim
        self.eps = eps  # 数值稳定性参数（避免除零）

        # 1. 3D全局池化：压缩深度(D)、高度(H)、宽度(W)三个空间维度
        self.global_pool = nn.AdaptiveAvgPool3d(1)  # 3D池化，输出形状为(B, C, 1, 1, 1)

        # 2. 特征投影与非线性变换（与2D版本一致，仅处理通道维度）
        self.projection = nn.Sequential(
            nn.Linear(in_channels, out_dim),  # 从输入通道数映射到out_dim
            nn.GELU(),                        # 非线性激活
            nn.Linear(out_dim, out_dim),      # 进一步特征转换
        )

    def forward(self, x):
        # 输入3D特征形状：(B, C, D, H, W)，其中D为深度，H为高度，W为宽度
        # 1. 3D全局池化：压缩所有空间维度（D, H, W → 1,1,1）
        x = self.global_pool(x)  # 输出形状：(B, C, 1, 1, 1)

        # 2. 移除多余的空间维度（挤压1维）
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # 连续挤压最后三个维度，输出：(B, C)

        # 3. 特征投影：从通道维度(C)映射到目标维度(out_dim)
        x = self.projection(x)  # 输出形状：(B, out_dim)

        # 4. 范数约束到[0,1]（原逻辑隐含的约束，显式化以确保效果）
        x_norm = torch.norm(x, dim=1, keepdim=True) + self.eps  # 计算每个样本的L2范数
        x = x / x_norm  # 归一化到单位球面上（范数=1）
        x = (x + 1) / 2  # 映射到[0,1]区间（利用对称性质：[-1,1]→[0,1]）

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

class ShallowAttentionFusion3D(nn.Module):
    """层级特征注意力融合模块（3D版本）：对两个同层级3D特征进行融合"""

    def __init__(self, in_channels):
        super().__init__()
        # 通道注意力：学习通道维度的权重（适配3D特征）
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 3D全局平均池化（对D、H、W维度池化）
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False),  # 3D卷积
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),  # 3D卷积
            nn.Sigmoid()
        )

        # 空间注意力：学习3D空间维度（D、H、W）的权重
        self.spatial_att = nn.Sequential(
            # 输入为两个特征的通道均值（维度2→1），3D卷积保持空间维度
            nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        # 融合后通道调整（3D卷积将2C降为C）
        self.fusion_conv = nn.Conv3d(in_channels * 2, in_channels, kernel_size=1, bias=False)

    def forward(self, feat1, feat2):
        # feat1和feat2为同层级3D特征，形状均为 (B, C, D, H, W)
        B, C, D, H, W = feat1.shape

        # 1. 通道注意力计算
        concat_feat = torch.cat([feat1, feat2], dim=1)  # 通道拼接：(B, 2C, D, H, W)
        channel_weight = self.channel_att(concat_feat)  # 输出：(B, C, 1, 1, 1)（对D、H、W维度广播）

        # 2. 空间注意力计算（基于两个特征的通道均值）
        feat1_mean = torch.mean(feat1, dim=1, keepdim=True)  # 通道均值：(B, 1, D, H, W)
        feat2_mean = torch.mean(feat2, dim=1, keepdim=True)  # 通道均值：(B, 1, D, H, W)
        spatial_input = torch.cat([feat1_mean, feat2_mean], dim=1)  # 拼接：(B, 2, D, H, W)
        spatial_weight = self.spatial_att(spatial_input)  # 输出：(B, 1, D, H, W)（对通道维度广播）

        # 3. 应用注意力权重（通道权重×空间权重）
        feat1_weighted = feat1 * channel_weight * spatial_weight  # 空间权重偏向feat1
        feat2_weighted = feat2 * channel_weight * (1 - spatial_weight)  # 空间权重互补偏向feat2

        # 4. 融合特征并调整通道数（从2C降为C）
        fused = torch.cat([feat1_weighted, feat2_weighted], dim=1)  # (B, 2C, D, H, W)
        fused = self.fusion_conv(fused)  # (B, C, D, H, W)，与原特征通道数一致

        return fused

class DeepAttentionFusion3D(nn.Module):
    """
    双模态3D特征融合模块（基于注意力机制）
    输入：两个3D特征图 x (B, C, D, H, W) 和 y (B, C, D, H, W)
    输出：融合后的3D特征图 (B, C, D, H, W)
    """
    def __init__(self, 
                 hidden_size,  # 特征通道数（输入x和y的通道数需与此一致）
                 num_heads=8,  # 注意力头数（需满足 hidden_size % num_heads == 0）
                 mlp_dim=2048,  # MLP中间层维度
                 dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size

        # 1. 注意力层（双模态独立的Q/K/V线性投影）
        self.query_x = nn.Linear(hidden_size, self.all_head_size)
        self.key_x = nn.Linear(hidden_size, self.all_head_size)
        self.value_x = nn.Linear(hidden_size, self.all_head_size)
        
        self.query_y = nn.Linear(hidden_size, self.all_head_size)
        self.key_y = nn.Linear(hidden_size, self.all_head_size)
        self.value_y = nn.Linear(hidden_size, self.all_head_size)
        
        # 注意力输出投影层
        self.out_x = nn.Linear(hidden_size, hidden_size)
        self.out_y = nn.Linear(hidden_size, hidden_size)

        # 2. 动态融合权重（可学习的参数，控制自注意力与跨模态注意力的权重）
        self.w11 = nn.Parameter(torch.tensor(0.5))  # x自注意力权重
        self.w12 = nn.Parameter(torch.tensor(0.5))  # x跨模态注意力权重
        self.w21 = nn.Parameter(torch.tensor(0.5))  # y自注意力权重
        self.w22 = nn.Parameter(torch.tensor(0.5))  # y跨模态注意力权重

        # 3. 层归一化与Dropout（稳定训练，防止过拟合）
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.proj_dropout = nn.Dropout(dropout_rate)
        self.norm_x1 = nn.LayerNorm(hidden_size)  # 注意力前的层归一化（x模态）
        self.norm_y1 = nn.LayerNorm(hidden_size)  # 注意力前的层归一化（y模态）
        self.norm_x2 = nn.LayerNorm(hidden_size)  # MLP前的层归一化（x模态）
        self.norm_y2 = nn.LayerNorm(hidden_size)  # MLP前的层归一化（y模态）

        # 4. MLP层（增强特征的非线性表达能力）
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
        """将特征拆分到多个注意力头，适配Transformer注意力计算格式"""
        # x输入形状：(B, N, all_head_size)，N为3D空间展平后的序列长度
        new_x_shape = x.size()[:-1] + (self.num_heads, self.attention_head_size)  # (B, N, num_heads, head_dim)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # 输出形状：(B, num_heads, N, head_dim)

    def forward(self, x, y):
        # -------------------------- 1. 3D特征→序列转换 --------------------------
        # 输入3D特征形状：(B, C, D, H, W)，其中C=hidden_size
        B, C, D, H, W = x.shape  # 解析3D特征的深度(D)、高度(H)、宽度(W)
        N = D * H * W  # 3D空间展平后的序列长度（原2D为H*W，3D新增深度维度D）
        
        # 展平逻辑：保留批量(B)和通道(C)，将3D空间维度(D,H,W)展平为序列维度(N)
        # 最终序列形状：(B, N, C)（适配Transformer注意力的输入格式）
        x_seq = x.flatten(2).transpose(1, 2)  # (B, C, D, H, W) → (B, C, N) → (B, N, C)
        y_seq = y.flatten(2).transpose(1, 2)  # 同x模态的展平逻辑

        # 残差连接备份（用于后续注意力和MLP阶段的残差相加）
        residual_x = x_seq
        residual_y = y_seq

        # -------------------------- 2. 注意力融合阶段 --------------------------
        # 注意力前的层归一化（Pre-LN结构，稳定训练）
        x_norm = self.norm_x1(x_seq)
        y_norm = self.norm_y1(y_seq)

        # 生成Q(查询)、K(键)、V(值)（双模态独立生成）
        qx = self.query_x(x_norm)
        kx = self.key_x(x_norm)
        vx = self.value_x(x_norm)
        
        qy = self.query_y(y_norm)
        ky = self.key_y(y_norm)
        vy = self.value_y(y_norm)

        # 拆分注意力头（适配多头注意力计算）
        qx = self.transpose_for_scores(qx)  # (B, num_heads, N, head_dim)
        kx = self.transpose_for_scores(kx)
        vx = self.transpose_for_scores(vx)
        
        qy = self.transpose_for_scores(qy)
        ky = self.transpose_for_scores(ky)
        vy = self.transpose_for_scores(vy)

        # 2.1 自注意力计算（模态内的注意力交互）
        # x模态自注意力
        attn_scores_x = torch.matmul(qx, kx.transpose(-1, -2)) / math.sqrt(self.attention_head_size)  # 缩放点积
        attn_probs_x = F.softmax(attn_scores_x, dim=-1)  # 注意力权重归一化
        attn_probs_x = self.attn_dropout(attn_probs_x)  # 注意力权重Dropout
        attn_output_x = torch.matmul(attn_probs_x, vx)  # 自注意力输出
        
        # y模态自注意力（同x模态逻辑）
        attn_scores_y = torch.matmul(qy, ky.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_y = F.softmax(attn_scores_y, dim=-1)
        attn_probs_y = self.attn_dropout(attn_probs_y)
        attn_output_y = torch.matmul(attn_probs_y, vy)

        # 2.2 跨模态注意力计算（模态间的注意力交互）
        # x模态查询 → y模态键值（用x的Q匹配y的K/V，捕捉x对y的依赖）
        attn_scores_cx = torch.matmul(qx, ky.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_cx = F.softmax(attn_scores_cx, dim=-1)
        attn_probs_cx = self.attn_dropout(attn_probs_cx)
        attn_output_cx = torch.matmul(attn_probs_cx, vy)
        
        # y模态查询 → x模态键值（用y的Q匹配x的K/V，捕捉y对x的依赖）
        attn_scores_cy = torch.matmul(qy, kx.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        attn_probs_cy = F.softmax(attn_scores_cy, dim=-1)
        attn_probs_cy = self.attn_dropout(attn_probs_cy)
        attn_output_cy = torch.matmul(attn_probs_cy, vx)

        # 2.3 注意力头拼接与投影（将多头输出合并为单通道维度）
        # x模态自注意力结果处理
        attn_output_x = attn_output_x.permute(0, 2, 1, 3).contiguous()  # (B, N, num_heads, head_dim)
        attn_output_x = attn_output_x.view(B, N, self.all_head_size)  # 拼接多头：(B, N, all_head_size)
        attn_output_x = self.out_x(attn_output_x)  # 投影回hidden_size维度
        attn_output_x = self.proj_dropout(attn_output_x)
        
        # y模态自注意力结果处理（同x模态）
        attn_output_y = attn_output_y.permute(0, 2, 1, 3).contiguous()
        attn_output_y = attn_output_y.view(B, N, self.all_head_size)
        attn_output_y = self.out_y(attn_output_y)
        attn_output_y = self.proj_dropout(attn_output_y)
        
        # x模态跨注意力结果处理（同自注意力）
        attn_output_cx = attn_output_cx.permute(0, 2, 1, 3).contiguous()
        attn_output_cx = attn_output_cx.view(B, N, self.all_head_size)
        attn_output_cx = self.out_x(attn_output_cx)
        attn_output_cx = self.proj_dropout(attn_output_cx)
        
        # y模态跨注意力结果处理（同自注意力）
        attn_output_cy = attn_output_cy.permute(0, 2, 1, 3).contiguous()
        attn_output_cy = attn_output_cy.view(B, N, self.all_head_size)
        attn_output_cy = self.out_y(attn_output_cy)
        attn_output_cy = self.proj_dropout(attn_output_cy)

        # 2.4 动态权重融合（自注意力+跨模态注意力）
        x_attn = self.w11 * attn_output_x + self.w12 * attn_output_cx  # x模态最终注意力特征
        y_attn = self.w21 * attn_output_y + self.w22 * attn_output_cy  # y模态最终注意力特征

        # 残差连接（注意力阶段）
        x_attn = x_attn + residual_x
        y_attn = y_attn + residual_y

        # -------------------------- 3. MLP增强阶段 --------------------------
        # MLP前的层归一化
        x_mlp = self.norm_x2(x_attn)
        y_mlp = self.norm_y2(y_attn)

        # MLP非线性变换（增强特征表达）
        x_mlp = self.mlp_x(x_mlp)
        y_mlp = self.mlp_y(y_mlp)

        # 残差连接（MLP阶段）
        x_final = x_mlp + x_attn  # x模态最终特征（注意力+MLP）
        y_final = y_mlp + y_attn  # y模态最终特征（注意力+MLP）

        # -------------------------- 4. 双模态融合与形状恢复 --------------------------
        # 双模态特征相加融合（可根据需求替换为concat+卷积等方式）
        fused_seq = x_final + y_final  # 融合后序列形状：(B, N, C)
        
        # 序列→3D特征恢复：将展平的序列还原为(B, C, D, H, W)
        fused_feat = fused_seq.transpose(1, 2).view(B, C, D, H, W)  # (B, N, C) → (B, C, N) → (B, C, D, H, W)
        
        return fused_feat

class HyperbolicAttentionFusion3D(nn.Module):
    """高效的双曲空间注意力融合模块（3D版本，完全向量化实现）"""

    def __init__(self, in_channels, hyper_c=0.2, hidden_dim=512):
        super().__init__()
        self.hyper_utils = HyperbolicUtils(c=hyper_c)
        self.in_channels = in_channels  # 输入特征通道数（需与3D特征通道一致）

        # 欧式特征→双曲空间的投影层（3D卷积，保持空间维度）
        self.to_hyperbolic1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),  # 3D 1x1卷积（不改变空间尺寸）
        )
        self.to_hyperbolic2 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
        )
        self.to_hyperbolic3 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
        )

        # 距离激活与局部距离编码（3D卷积适配）
        self.distance_activation = nn.Sigmoid()
        self.distance_mlp = nn.Sequential(
            nn.Conv3d(2, 2, kernel_size=1),  # 3D 1x1卷积（处理3D距离图）
        )

        # 最终特征融合FFN（3D卷积，压缩通道）
        self.ff = nn.Sequential(
            nn.Conv3d(2 * in_channels, hidden_dim, kernel_size=1),  # 2C→hidden_dim
            nn.GELU(),
            nn.Conv3d(hidden_dim, in_channels, kernel_size=1),      # hidden_dim→C（恢复原通道数）
        )

    def hyper_distance_map(self, u, v):
        """
        3D向量化计算双曲空间中两点u和v的测地距离图（保角缩放+数值稳定）
        参数:
            u: (B, C, D, H, W) 3D查询点特征
            v: (B, C, D, H, W) 3D目标点特征
        返回:
            distance_map: (B, 1, D, H, W) 3D距离图（每个空间位置对应一个距离值）
        """
        B, C, D, H, W = u.shape
        eps = 1e-6
        c = self.hyper_utils.c
        max_norm = (1.0 / math.sqrt(c)) - eps  # 双曲空间合法范数上限（防止溢出）

        # -------------------------- 1. 3D特征重塑与保角范数缩放 --------------------------
        # 维度转换：(B, C, D, H, W) → (B, D, H, W, C)（空间维度在前，通道在后，便于逐位置处理）
        u_reshaped = u.permute(0, 2, 3, 4, 1)  # 3D空间维度(D,H,W)展平前的重塑
        v_reshaped = v.permute(0, 2, 3, 4, 1)

        # 逐3D空间位置计算范数（每个(D,H,W)位置对应一个C维向量的范数）
        norm_u_original = torch.norm(u_reshaped, dim=4, keepdim=True) + eps  # (B, D, H, W, 1)
        norm_v_original = torch.norm(v_reshaped, dim=4, keepdim=True) + eps

        # 保角缩放：将范数约束在max_norm内（保持向量夹角不变，仅缩放长度）
        target_norm_u = torch.min(norm_u_original, torch.full_like(norm_u_original, max_norm))
        target_norm_v = torch.min(norm_v_original, torch.full_like(norm_v_original, max_norm))
        scale_u = target_norm_u / norm_u_original
        scale_v = target_norm_v / norm_v_original

        u_scaled = u_reshaped * scale_u  # (B, D, H, W, C)：范数合规的3D特征
        v_scaled = v_reshaped * scale_v

        # -------------------------- 2. 3D双曲测地距离计算 --------------------------
        # 计算缩放后的范数（加钳位进一步确保数值稳定）
        norm_u = torch.norm(u_scaled, dim=4, keepdim=True).clamp(max=max_norm)  # (B, D, H, W, 1)
        norm_v = torch.norm(v_scaled, dim=4, keepdim=True).clamp(max=max_norm)

        # 逐3D位置点积与差向量平方和
        uv_dot = torch.sum(u_scaled * v_scaled, dim=4, keepdim=True)  # (B, D, H, W, 1)
        diff_norm_sq = torch.sum((u_scaled - v_scaled) ** 2, dim=4, keepdim=True)  # (B, D, H, W, 1)

        # 庞加莱圆盘测地距离公式（完全向量化，适配3D所有空间位置）
        numerator = 1 + 2 * c * diff_norm_sq
        denominator = (1 - c * norm_u ** 2) * (1 - c * norm_v ** 2) + eps
        inside_acosh = (numerator / denominator).clamp(min=1 + eps)  # 确保acosh输入≥1（数学性质要求）

        # 计算距离并重塑为3D特征格式
        distance = (1 / math.sqrt(c)) * torch.acosh(inside_acosh)  # (B, D, H, W, 1)
        distance_map = distance.permute(0, 4, 1, 2, 3)  # (B, 1, D, H, W)：恢复通道优先格式

        return distance_map

    def forward(self, euclidean_fused, feat1, feat2):
        """
        3D双曲注意力融合前向传播
        参数:
            euclidean_fused: 欧式空间预融合特征 (B, C, D, H, W)
            feat1: 模态1 3D特征 (B, C, D, H, W)
            feat2: 模态2 3D特征 (B, C, D, H, W)
        返回:
            fused_feature: 最终融合3D特征 (B, C, D, H, W)
        """
        B, C, D, H, W = euclidean_fused.shape  # 解析3D特征维度
        spatial_flat_dim = D * H * W  # 3D空间展平后的总长度（替代2D的H*W）

        # -------------------------- 1. 欧式特征→3D双曲空间映射 --------------------------
        # 预融合特征映射（euclidean_fused → 双曲空间查询点）
        query_hyper = self.to_hyperbolic1(euclidean_fused)  # (B, C, D, H, W)
        # 展平3D空间维度：(B, C, D, H, W) → (B, C, spatial_flat_dim) → (B*spatial_flat_dim, C)
        query_hyper_flat = query_hyper.permute(0, 2, 3, 4, 1).reshape(-1, self.in_channels)
        # 双曲映射后恢复3D格式：(B*spatial_flat_dim, C) → (B, D, H, W, C) → (B, C, D, H, W)
        query_hyper = self.hyper_utils.euclid_to_hyper(query_hyper_flat)
        query_hyper = query_hyper.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # 模态1特征映射（同query逻辑）
        feat1_hyper = self.to_hyperbolic2(feat1)
        feat1_hyper_flat = feat1_hyper.permute(0, 2, 3, 4, 1).reshape(-1, self.in_channels)
        feat1_hyper = self.hyper_utils.euclid_to_hyper(feat1_hyper_flat)
        feat1_hyper = feat1_hyper.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # 模态2特征映射（同query逻辑）
        feat2_hyper = self.to_hyperbolic3(feat2)
        feat2_hyper_flat = feat2_hyper.permute(0, 2, 3, 4, 1).reshape(-1, self.in_channels)
        feat2_hyper = self.hyper_utils.euclid_to_hyper(feat2_hyper_flat)
        feat2_hyper = feat2_hyper.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)

        # -------------------------- 2. 3D双曲距离计算与注意力权重 --------------------------
        # 计算查询点与两个模态的3D距离图
        dist1_map = self.hyper_distance_map(query_hyper, feat1_hyper)  # (B, 1, D, H, W)
        dist2_map = self.hyper_distance_map(query_hyper, feat2_hyper)  # (B, 1, D, H, W)

        # 相对距离比例（避免除零）
        dist_sum = dist1_map + dist2_map + 1e-8
        rel_dist1 = dist1_map / dist_sum  # 距离越大，该模态权重应越小
        rel_dist2 = dist2_map / dist_sum

        # 3D空间感知的距离编码与权重归一化
        weights = torch.cat([-rel_dist1, -rel_dist2], dim=1)  # (B, 2, D, H, W)：负距离→距离小权重高
        weights = self.distance_mlp(weights)  # 3D卷积捕捉局部距离依赖
        normalized_weights = torch.softmax(weights, dim=1)  # 通道维度归一化（2个模态权重和为1）
        weight1, weight2 = normalized_weights[:, 0:1, ...], normalized_weights[:, 1:2, ...]  # 各(B,1,D,H,W)

        # -------------------------- 3. 双曲切空间加权融合 --------------------------
        # 双曲特征→欧式切空间（切空间支持线性加权，避免双曲空间加法的复杂性）
        feat1_tangent = self.hyper_utils.hyper_to_euclid(feat1_hyper_flat)  # (B*spatial_flat_dim, C)
        feat2_tangent = self.hyper_utils.hyper_to_euclid(feat2_hyper_flat)  # (B*spatial_flat_dim, C)

        # 3D注意力权重展平（匹配切空间特征形状）
        weight1_flat = weight1.permute(0, 2, 3, 4, 1).reshape(-1, 1)  # (B*spatial_flat_dim, 1)
        weight2_flat = weight2.permute(0, 2, 3, 4, 1).reshape(-1, 1)  # (B*spatial_flat_dim, 1)

        # 切空间线性加权（向量化计算，效率高）
        fused_tangent = weight1_flat * feat1_tangent + weight2_flat * feat2_tangent  # (B*spatial_flat_dim, C)

        # -------------------------- 4. 特征恢复与最终融合 --------------------------
        # 切空间特征恢复为3D格式
        fused_tangent = fused_tangent.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)

        # 3D LayerNorm：对齐与欧式预融合特征的数值范围（归一化通道+3D空间）
        if not hasattr(self, "tangent_norm"):
            self.tangent_norm = nn.LayerNorm([C, D, H, W], device=fused_tangent.device)
        fused_euclidean = self.tangent_norm(fused_tangent)

        # 双曲融合特征 + 欧式预融合特征：通道拼接后压缩回原通道数
        fused_feature = torch.cat([fused_euclidean, euclidean_fused], dim=1)  # (B, 2C, D, H, W)
        fused_feature = self.ff(fused_feature)  # (B, C, D, H, W)

        return fused_feature
    

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
                 in_channels=3, classes=4, hyper_c=0.5,feature_dim=512):
        super().__init__()
        
        # 编码器1和编码器2
        self.model1 = model_3Dunet.UNet()
        self.model2 = model_3Dunet.UNet()

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
            ShallowAttentionFusion3D(enc_ch) for enc_ch in encoder_channels
        ])

        self.mbafusion = DeepAttentionFusion3D(hidden_size =encoder_channels[-1])

        self.hyperbolic_fusion_blocks = nn.ModuleList([
            HyperbolicAttentionFusion3D(enc_ch, hyper_c) for enc_ch in encoder_channels
        ])

        # 添加特征压缩模块：将各层级特征统一到(B, feature_dim)
        self.feature_compressors1 = nn.ModuleList([
            FeatureCompressor3D(enc_ch, feature_dim) for enc_ch in encoder_channels
        ])
        self.feature_compressors2 = nn.ModuleList([
            FeatureCompressor3D(enc_ch, feature_dim) for enc_ch in encoder_channels
        ])
        self.feature_compressors_fused = nn.ModuleList([
            FeatureCompressor3D(enc_ch, feature_dim) for enc_ch in encoder_channels
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
            B, C, D, H, W = feat_map.shape

            # 压缩向量融合（取fused的压缩特征，或1+2的均值，增强全局信息）
            global_vec = (compressed1[i] + compressed2[i] + compressed_fused[i]) / 3  # (B, 512)
            # 向量→特征图：先线性映射到通道数C，再扩展空间维度（B, C）→ (B, C, 1, 1) → (B, C, D, H, W)
            global_feat = self.compress_injectors[i](global_vec).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1, 1)
            global_feat = global_feat.expand(-1, -1, D, H, W)  # (B, C, D, H, W)

            # 融合全局信息与局部特征图（元素相加或卷积融合）
            fused_with_global = feat_map + global_feat  # 简单高效，适合初期
            decoder_inputs.append(fused_with_global)


        # 5. 融合后的特征传入解码器
        decoder_output = self.decoder(decoder_inputs)


        # 6. 分割头输出结果
        mask = self.segmentation_head(decoder_output)

        geom_loss = self.geometry_loss_fn(hyper_fused, hyper_features1, hyper_features2)

        # 推理模式下只返回掩码和双曲特征
        return mask, geom_loss

# 测试模型
if __name__ == "__main__":
    # 创建模型
    model = DualInputHyperbolicUNet()

    # 创建测试输入和标签
    x1 = torch.randn(2, 1, 155, 240, 240)
    x2 = torch.randn(2, 1, 155, 240, 240)
    labels = torch.randint(0, 4, (2, 1, 155, 240, 240))  # 随机标签

    # 训练模式下前向传播
    output, geo_loss = model(x1, x2)

    # # 推理模式下前向传播
    # with torch.no_grad():
    #     output, geo_loss = model(x1, x2)
    #     print(f": {output}")
    #     print(f": {geo_loss}")

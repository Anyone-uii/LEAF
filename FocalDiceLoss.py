import torch
import segmentation_models_pytorch as smp


class FocalDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=1.0, focal_weight=1.2):
        """
        FocalDiceLoss combines Focal Loss and Dice Loss for multi-class semantic segmentation.
        :param alpha: balancing factor for Focal Loss (default 0.25)
        :param gamma: focusing parameter for Focal Loss (default 2.0)
        :param dice_weight: weight for Dice Loss contribution
        :param focal_weight: weight for Focal Loss contribution
        """
        super(FocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass', alpha=alpha, gamma=gamma)
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')

    def forward(self, preds, labels):
        """
        Compute the combined Focal and Dice Loss.
        :param preds: Predictions, shape (B, num_classes, H, W)
        :param labels: Ground truth labels, shape (B, H, W)
        :return: Combined loss value
        """
        # Calculate individual losses
        dice = self.dice_loss(preds, labels)
        focal = self.focal_loss(preds, labels)

        # Return combined loss
        return self.dice_weight * dice + self.focal_weight * focal

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, inputs, targets):
        # 平滑因子，防止除零错误
        smooth = 1.0
        # 对输入进行 sigmoid 操作，将其转换为概率
        inputs = torch.sigmoid(inputs)
        # 将输入和目标张量展平，方便计算
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # 计算真阳性
        tp = (inputs * targets).sum()
        # 计算假阴性
        fn = (targets * (1 - inputs)).sum()
        # 计算假阳性
        fp = ((1 - targets) * inputs).sum()
        # 计算 Tversky 系数
        tversky = (tp + smooth) / (tp + self.alpha * fn + self.beta * fp + smooth)
        # 计算 Focal Tversky 损失
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky
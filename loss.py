# loss.py

"""
自定义的自适应加权损失函数模块。
"""

import torch
import torch.nn as nn

class AdaptiveOpticalLoss(nn.Module):
    """
    根据每个损失分量的不确定性自适应地学习权重的损失函数。
    """
    def __init__(self, num_losses: int = 3):
        super().__init__()
        # 为每个损失分量创建一个可学习的对数方差
        self.log_vars = nn.Parameter(torch.zeros(num_losses))

    def forward(self, *losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算加权后的总损失。

        Args:
            *losses: 多个损失张量。

        Returns:
            一个元组，包含 (总损失, 各分量的权重)。
        """
        losses_tensor = torch.stack(losses)
        
        # 基于不确定性的加权： precision = exp(-log_var)
        precision = torch.exp(-self.log_vars)
        
        # 总损失 = Σ (precision_i * loss_i + log_var_i)
        weighted_loss = torch.sum(precision * losses_tensor + self.log_vars)
        
        # 计算归一化权重，用于日志记录或分析
        weights = precision / precision.sum()
        
        return weighted_loss, weights.detach()
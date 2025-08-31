import torch
import torch.nn as nn

# 假设 FocalLoss 和 MaskedNLLLoss 是自定义的损失函数
from model_cm import FocalLoss, MaskedNLLLoss

def initialize_loss_function(args, cuda):
    """
    根据数据集和配置初始化损失函数。
    """
    # 根据数据集设置损失权重
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([
            1/0.086747, 1/0.144406, 1/0.227883,
            1/0.160585, 1/0.127711, 1/0.252668
        ])
    else:
        loss_weights = None

    # 选择和初始化损失函数
    if args.Dataset == 'MELD':
        loss_function = FocalLoss()
    else:
        if args.class_weight and loss_weights is not None:
            # 根据模型类型选择损失函数
            if args.graph_model:
                loss_function = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
            else:
                loss_function = MaskedNLLLoss(loss_weights.cuda() if cuda else loss_weights)
        else:
            if args.graph_model:
                loss_function = nn.NLLLoss()
            else:
                loss_function = MaskedNLLLoss()

    return loss_function
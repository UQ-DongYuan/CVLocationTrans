import torch
import torch.nn.functional as F
import numpy as np


def cross_entropy_loss(pred, target):
    """
    :param pred: [B, N=1024, 1] output for patch/location prediction
    :param target: label list [(index, ty, tx)], len(target) = B
    :return: loss value
    """
    pred_input = pred.squeeze(2)  # [B N]
    label_index = torch.zeros(pred_input.shape[0], dtype=torch.long)  # [B]
    for i in range(len(target)):
        label_index[i] = target[i][0]
    loss = F.cross_entropy(pred_input, label_index.to('cuda'))
    return loss

def regression_loss(pred, target):
    """
    :param pred: [B, N=1024, 2] prediction for ty and tx
    :param target: label list [(index, ty, tx)], len(target) = B
    :return: regression loss
    """
    label_txy = torch.zeros(pred.shape[0], 2).to('cuda')  # [B, 2]
    pred_input = torch.zeros(pred.shape[0], 2).to('cuda')  # [B, 2]
    for i in range(len(target)):
        idx, ty, tx = target[i]
        label_txy[i] = torch.tensor([ty, tx])
        pred_input[i] = pred[i, idx, :]
    loss = F.mse_loss(pred_input, label_txy)
    return loss

import torch
from torch import nn
import torch.nn.functional as F

def Loss_multi_discriminator(label_clean, label_recon):
    loss = 0
    for ii in range(len(label_recon)):
        prediction_clean = label_clean[ii][0]
        prediction_recon = label_recon[ii][0]
        loss_clean = nn.MSELoss()(prediction_clean, torch.ones_like(prediction_clean))
        loss_recon = nn.MSELoss()(prediction_recon, torch.zeros_like(prediction_recon))
        loss_ii = loss_clean + loss_recon
        loss += loss_ii
    loss /= len(label_clean)
    return loss



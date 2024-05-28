import torch
import torch.nn as nn
import numpy as np


class loss_MTFAA_t(nn.Module):
    def __init__(self):
        super(loss_MTFAA_t, self).__init__()
        self.WINDOW = torch.sqrt(torch.hann_window(512) + 1e-8)

    def forward(self,y_pred, y_true):
        device = y_true.device
        
        pred_stft=torch.stft(y_pred, 512, 256, 512, self.WINDOW.to(device),return_complex=False)
        true_stft=torch.stft(y_true, 512, 256, 512, self.WINDOW.to(device),return_complex=False)
        #print(pred_stft.shape,true_stft.shape)#[B, 257, 126, 2]
        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = torch.mean((pred_real_c - true_real_c)**2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
        mag_loss = torch.mean((pred_mag**(0.3)-true_mag**(0.3))**2)

        # sisnr这里没问题，应该是0.1*10*torch.log10()
        #y_pred = torch.istft(pred_stft, 512, 256, 512, self.WINDOW.to(device))
        #y_true = torch.istft(true_stft, 512, 256, 512, self.WINDOW.to(device))
        #snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),(torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        #snr_loss = 10 * torch.log10(snr + 1e-7)
        y_true = torch.sum(y_true * y_pred, dim=-1, keepdim=True) * y_true / (torch.sum(torch.square(y_true),dim=-1,keepdim=True) + 1e-8)
        sisnr =  - 0.5 * torch.log10(torch.sum(torch.square(y_true),dim=-1,keepdim=True) / torch.sum(torch.square(y_pred - y_true),dim=-1,keepdim=True) + 1e-8)
        
        return 30*(real_loss + imag_loss) + 70*mag_loss + torch.mean(sisnr)

    
class loss_MTFAA_t_f_frameshift(nn.Module):
    def __init__(self):
        super(loss_MTFAA_t_f_frameshift, self).__init__()
        self.loss = loss_MTFAA_t()

    def forward(self,y_pred, y_true):
        loss1 = self.loss(y_pred, y_true)
        loss2 = self.loss(y_pred[:,:-64], y_true[:,64:])
        loss3 = self.loss(y_pred[:,64:], y_true[:,:-64])
        loss4 = self.loss(y_pred[:,:-128], y_true[:,128:])
        loss5 = self.loss(y_pred[:,128:], y_true[:,:-128])
        
        loss = min(loss1, loss2, loss3, loss4, loss5)
        
        return loss



if __name__=='__main__':
    a = torch.randn(2,10000)
    b = torch.randn(2,9990)
    loss = loss_si_snr()
    mse = loss(a,b)
    
    S_ = torch.randn(3, 257, 91, 2)
    S = torch.randn(3, 257, 91, 2)
    Loss = loss_cRIMag()
    loss = Loss(S_, S)
    print(loss)
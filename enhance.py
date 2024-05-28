import torch
torch.backends.cudnn.enabled = False
import numpy as np
import soundfile as sf
from Lite.causalwAtt import TFGridNet128_hs4 as TFGridNet
import os
import torchaudio as audio
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

n_mics = 1 ###
fs = 16000
batch_len = 3 * fs # each batch keeps 3s
pad = 3 * fs # batch padding is 3s before and after
is_single_checkpoint = True

if n_mics==1:

    noisypath = 
    filelist=os.listdir(noisypath)
    exppath=
    testgenerpath = 
    device  = torch.device("cuda:0")
    model = TFGridNet().to(device)#########
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    if is_single_checkpoint:
        ckpoints='checkpoints/model_0006.tar'
        checkpoint = torch.load(exppath+ckpoints, map_location=device)#+'checkpoints/'
        for k, v in checkpoint['model'].items():
            if '.' in k:
                name = k#[7:] # remove `module.`
                new_state_dict[name] = v 


    model.load_state_dict(new_state_dict,strict=False)#checkpoint['model'])#(checkpoint['model'])# g6 dont change
    model.eval()

    with torch.no_grad():
        for i in tqdm(range(len(filelist))):
            if '.wav' in filelist[i]:
                noisy, f_s= audio.load(noisypath+filelist[i], frame_offset=0 , num_frames=-1, normalize=True, channels_first=True)
                noisy = noisy[:1,:]
                if f_s!=fs:noisy=audio.transforms.Resample(orig_freq=f_s, new_freq=fs)(noisy)
                noisy = noisy[:,:].to(device)############

                n_len = noisy.shape[1]

                if n_len <= fs*10:# more than 10s
                    esti_tagt = model(noisy)# [1, 16000]
                    enhanced = esti_tagt[0][-1].squeeze().cpu().numpy()
                    
                else:
                    batch_index = 0
                    enh_patch = torch.zeros([1,1])
                    while(enh_patch.shape[1]<n_len):
                        #print(enh_patch.shape)
                        if batch_index==0:
                            noisy_batch = noisy[:,:batch_len+pad]
                            esti_tagt = model(noisy_batch)# [1, 16000]
                            enhanced = esti_tagt[0][0]
                            enh_patch = enhanced[:,:batch_len]
                            # print('1')
                        elif (batch_index+1)*batch_len + pad >= n_len:
                            noisy_batch = noisy[:,batch_index*batch_len - pad:]
                            #print(noisy_batch.shape,noisy.shape,batch_index,  batch_index*batch_len - pad)
                            esti_tagt = model(noisy_batch)# [1, 16000]
                            enhanced = esti_tagt[0][0]
                            enh_patch = torch.cat([enh_patch, enhanced[:,pad:]], dim=1)
                            # print('2')
                        else:
                            noisy_batch = noisy[:,batch_index*batch_len - pad:(batch_index+1)*batch_len + pad]
                            esti_tagt = model(noisy_batch)# [1, 16000]
                            enhanced = esti_tagt[0][0]
                            enh_patch = torch.cat([enh_patch, enhanced[:,pad:pad+batch_len]], dim=1)
                            # print('3')
                        batch_index = batch_index + 1
                    enhanced = enh_patch.squeeze().cpu().numpy()

                os.makedirs(exppath + testgenerpath, exist_ok=True)
                sf.write(exppath+ testgenerpath +filelist[i],enhanced.T, 16000)


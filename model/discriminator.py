import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_disc(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(conv_disc, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, (3,3), stride=stride)
        self.conv = torch.nn.utils.weight_norm(self.conv)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        x = self.lrelu(self.conv(x))
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_blocks = nn.Sequential(conv_disc(2, 8, 2),
                                         conv_disc(8, 16, 2),
                                         conv_disc(16, 32, 2),
                                         conv_disc(32, 64, 2),
                                         conv_disc(64, 128, 1),
                                         conv_disc(128, 256, 1),
                                         nn.Conv2d(256, 1, 1))
        self.act_last = nn.Sigmoid()
    def forward(self, x):
        feature_list = []
        for i, conv_block in enumerate(self.conv_blocks):
            #print(i)
            #print(x.size())
            x = conv_block(x)
            if i < len(self.conv_blocks):
                #print(len(self.conv_blocks))
                feature_list.append(x)
        x = self.act_last(x)
        #print(x.size())
        return x, feature_list
    
    
class MultiResolutionDiscriminator(nn.Module):
    def __init__(self,
                 fft_sizes=[2048, 1024, 512],
                hop_sizes=[240, 120, 50],
                win_lengths=[1200, 600, 240]):
        super(MultiResolutionDiscriminator, self).__init__()
        self.n_resolution = len(fft_sizes)
        self.discriminators = torch.nn.ModuleList()
        self.fft_sizes = fft_sizes 
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        for _ in fft_sizes:
            self.discriminators += [Discriminator()]
    def forward(self, x):
        ''' 
        x: time domain signal, (bs, T)
        '''
        output = []
        for fs, ss, wl, model in zip(self.fft_sizes, self.hop_sizes, self.win_lengths, self.discriminators):
            x_spec = torch.stft(x, fs, ss, wl, return_complex=False) #(bs, F, T, 2)
            x_spec = x_spec.permute(0, 3, 2, 1).contiguous() #(bs, 2, T, F)
            x_output = model(x_spec)
            output += [x_output]
            #print(len(output))
        return output
    
    
class MultiBandDiscriminator(nn.Module):
    def __init__(self, n_band = 3):
        super(MultiBandDiscriminator, self).__init__()
        self.n_band  = n_band 
        self.discriminators = torch.nn.ModuleList()
        for _ in range(n_band):
            self.discriminators += [Discriminator()]
    def forward(self, x):
        ''' 
        x: spectrogram domain signal, (bs, 2*n, T, F)
        '''
        output = []
        for i, model in enumerate(self.discriminators):
            x_subband = x[:, i*2:(i+1)*2, :, :].contiguous()
            x_output = model(x_subband)
            output += [x_output]
        return output
    
    
class MultiDiscriminator(nn.Module):
    def __init__(self, 
                 n_band = 3, 
                 fft_sizes=[2048, 1024, 512],
                hop_sizes=[240, 120, 50],
                win_lengths=[1200, 600, 240]):
        super(MultiDiscriminator, self).__init__()
        self.n_band  = n_band 
        self.mrd = MultiResolutionDiscriminator(fft_sizes, hop_sizes, win_lengths)
        # self.mbd = MultiBandDiscriminator(n_band)
    def forward(self, x_time):
        output_mrd = self.mrd(x_time)
        # output_mbd = self.mbd(x_spec)
        return output_mrd
        
if __name__ == "__main__":
    
    model = MultiDiscriminator()
    B = 1
    x = torch.randn(B, 180000)#51400-256)  # (B, N,M)
    s = model(x)
    print(len(s),s[0][0].shape,s[1][0].shape,s[2][0].shape)
    '''
    # para/cplx
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(model , (160000,), as_strings=True,print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    from deepspeed.profiling.flops_profiler import get_model_profile
    from deepspeed.accelerator import get_accelerator

    with get_accelerator().device(1):
        #model = nn.GRU(10, 20, 2)
        #model = TFGridNet128()
        flops, macs, params = get_model_profile(model=model, # model
                                        input_shape=(1, 320000), # input shape to the model. If specified, the model takes a tensor with this shape as the only positional argument.
                                        args=None, # list of positional arguments to the model.
                                        kwargs=None, # dictionary of keyword arguments to the model.
                                        print_profile=True, # prints the model graph with the measured profile attached to each module
                                        detailed=False, # print the detailed profile
                                        module_depth=-1, # depth into the nested modules, with -1 being the inner most modules
                                        top_modules=1, # the number of top modules to print aggregated profile
                                        warm_up=10, # the number of warm-ups before measuring the time of each module
                                        as_string=False, # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
                                        output_file=None, # path to the output file. If None, the profiler prints to stdout.
                                        ignore_modules=None) # the list of modules to ignore in the profiling
    '''
        
'''
multi gpu version
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1" 
import toml
import torch
import argparse
import torch.distributed as dist
from collections import OrderedDict
from trainer_mul import Trainer
from model.causalTFGridNet import TFGridNet128_hs4 as TFGridNet
from Dataloader import Dataset
from binaural_loss import loss_MTFAA_t_f_frameshift
from model.discriminator import MultiDiscriminator

def run(rank,config,args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12361'
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    torch.cuda.set_device(rank)
    dist.barrier()

    args.rank = rank
    args.device = torch.device(rank)

    train_dataset = Dataset(**config['train_dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, sampler=train_sampler,
                                                    **config['train_dataloader'])
    
    validation_dataset = Dataset(**config['validation_dataset'])
    validation_sampler = torch.utils.data.distributed.DistributedSampler(validation_dataset)
    validation_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset, sampler=validation_sampler,
                                                        **config['validation_dataloader'])

    model = TFGridNet()
    discriminator = MultiDiscriminator()
    
    model.to(args.device)
    discriminator.to(args.device)

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    discriminator = torch.nn.parallel.DistributedDataParallel(discriminator, device_ids=[rank])

    #optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['lr'])
    optimizer = NoamOpt(model_size=config['network_config']['model_size'], factor=1.0, warmup=10000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))#一般warmup=step左右
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

    if config['loss']['loss_func'] == 'MTFAA':
        loss = loss_MTFAA_t_f_frameshift()
    elif config['loss']['loss_func'] == 'wavmag':
        loss = loss_wavmag()
    else:
        raise NotImplementedError

    trainer = Trainer(config=config, model=[model,discriminator],optimizer=[optimizer,optimizer_dis], loss_func=loss,
                      train_dataloader=train_dataloader, validation_dataloader=validation_dataloader, 
                      train_sampler=train_sampler,args=args)

    trainer.train()
    dist.destroy_process_group()

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 100
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        if self._step == 200000000:
            self._step = 1
            #self.factor = self.factor * 0.9
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))#step = warmnp时，两个相等

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', '--config', default='den_DNS3_discri.toml')

    args = parser.parse_args()

    config = toml.load(args.config)
    args.world_size = config['DDP']['world_size']
    torch.multiprocessing.spawn(
        run, args=(config, args,), nprocs=args.world_size, join=True)

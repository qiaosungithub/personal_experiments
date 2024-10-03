import os
log_file = './log.log'
with open(log_file,'w') as f:
    # write the result of !nvidia-smi
    with os.popen('nvidia-smi') as p:
        f.write(p.read())


try:
    from ViT.train import train
    from ViT.utils import cifar_train_set, cifar_val_set
    from ViT.model import *
except:
    from personal_experiments.ViT.ViT_vanilla.train_zhh import train
    from utils import cifar_train_set, cifar_val_set
    from model import *

from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import datetime

train_loader = DataLoader(cifar_train_set, 256, shuffle=True, drop_last=False, pin_memory=True)
val_loader = DataLoader(cifar_val_set, 500, shuffle=True, drop_last=False, pin_memory=True)

import os
if not os.path.exists("ViT/log"):
    os.makedirs("ViT/log")

def timestr():
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")

def get_outdir(time_str):
    outdir = f"ViT/log/{time_str}.out"
    return outdir

# Here is the hyperparameters

epochs = 50
patch_size = 8
embed_dim = 256
n_layers = 6
heads = 8
attn_dim = 512
mlp_dim = None # default to 4*embed_dim
pool = 'cls'
dropout = 0.0

model = ViT(image_size=32, patch_size=patch_size, num_classes=10, embed_dim=embed_dim, n_layers=n_layers, heads=heads, attn_dim=attn_dim, mlp_dim=mlp_dim, pool=pool, dropout=dropout)

optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), weight_decay=5e-4)

time_str = timestr()

print(f"Time string: {time_str}")

# print the model and the number of parameters
# print(model.transformer)
print(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

train(epochs=epochs, model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(), 
      train_loader=train_loader, val_loader=val_loader, outdir=get_outdir(time_str))

# save model
torch.save(model.state_dict(), f"ViT/models/{time_str}.pth")
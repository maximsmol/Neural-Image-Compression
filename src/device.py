import torch

from cli import args

use_cuda = torch.cuda.is_available() and not args.no_cuda
print(f'CUDA found={torch.cuda.is_available()}')
print(f'CUDA={use_cuda}, N GPUs={torch.cuda.device_count()}')
device = torch.device('cuda' if use_cuda else 'cpu')

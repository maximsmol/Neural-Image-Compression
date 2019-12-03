import torch

from cli import args

use_cuda = False

if torch.cuda.is_available():
  print('[V] CUDA available')
  if not args.no_cuda:
    print(f'   [V] Using {torch.cuda.device_count()} GPUs')
    use_cuda = True
  else:
    print('   [X] Disabled by command line flag')
else:
  print('[X] CUDA not available')

device = torch.device('cuda' if use_cuda else 'cpu')

print(f'Selected device: {device}')

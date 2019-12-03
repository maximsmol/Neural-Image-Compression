import torch
import numpy as np

from device import use_cuda

torch.manual_seed(0)
if use_cuda:
  torch.cuda.manual_seed(0)
np.random.seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

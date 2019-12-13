import torch
import torch.nn as nn

class ProbabilityModel(nn.Module):
  def __init__(self):
    super(ProbabilityModel, self).__init__()

    self.probs = nn.Parameter(torch.Tensor(1, 96, 1, 1))
    nn.init.constant_(self.probs, .5)

  def likelihood(self, x):
    return -torch.where(
      x > .5,
      self.probs,
      1 - self.probs
    ).log2().sum()

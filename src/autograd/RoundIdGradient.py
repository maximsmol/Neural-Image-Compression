import torch

class RoundIdGradient(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x):
    return x.round()

  @staticmethod
  def backward(ctx, g):
    return g

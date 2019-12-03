import torch

class ByteClampIdGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clamp(0, 1)

    @staticmethod
    def backward(ctx, g):
        return g

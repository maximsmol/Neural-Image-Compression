import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Normalize

from device import device

train_stats = pickle.load(open('data/train_stats.pickle', 'rb'))

means_tensor = torch.tensor([train_stats['means']['r'], train_stats['means']['g'], train_stats['means']['b']]).to(device)
std_tensor = torch.tensor([train_stats['std']['r'], train_stats['std']['g'], train_stats['std']['b']]).to(device)

class ByteClampIdGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clamp(0, 1)

    @staticmethod
    def backward(ctx, g):
        return g

class RoundIdGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g

class Residual(nn.Module):
  def __init__(self):
    super(Residual, self).__init__()

    self.convs = nn.ModuleList([
      nn.Conv2d(128, 128, (3, 3), padding=1), # fixme: should this always be padded? 0-padded?
      nn.Conv2d(128, 128, (3, 3), padding=1)
    ])
    self.bns = nn.ModuleList([
      nn.BatchNorm2d(128),
      nn.BatchNorm2d(128)
    ])

  def forward(self, x):
    identity = x
    for i in range(len(self.convs)):
      x = self.bns[i](x)
      x = F.relu(x)
      x = self.convs[i](x)
    x = x + identity
    return x

class Net(nn.Module):
  def __init__(self):
    super(Net, self).__init__()

    # Afterwards, the image is convolved and spatially downsampled while at the same time increasing the number of channels to 128.
    self.encoder_entry = nn.Sequential(
        nn.Conv2d(3, 64, (5, 5), stride=2),
        nn.Conv2d(64, 128, (5, 5), stride=2)
      )
    # This is followed by three residual blocks (He et al., 2015),
    #   where each block consists of an additional two convolutional layers with 128 filters each.
    self.encoder_residuals = nn.ModuleList([
      Residual(),
      Residual(),
      Residual()
    ])
    # A final convolutional layer is applied and the coefficients downsampled again before quantization through rounding to the nearest integer.
    self.encoder_exit = nn.Conv2d(128, 96, (5, 5), stride=2)

    nn.init.normal_(self.encoder_exit.weight, mean=100., std=10.)
    self.encoder_exit.weight.data *= (torch.rand(self.encoder_exit.weight.data.shape).round()*2 - 1)

    # The decoder mirrors the architecture of the encoder (Figure 9).
    #   Instead of mirror-padding and valid convolutions, we use zero-padded convolutions.

    # Upsampling is achieved through convolution followed by a reorganization of the coefficients.
    #   This reorganization turns a tensor with many channels into a tensor of the same dimensionality but with fewer channels and larger spatial extent (for details, see Shi et al., 2016).
    # A convolution and reorganization of coefficients together form a sub-pixel convolution layer.
    decoder_entry_conv = nn.Conv2d(96, 512, (3, 3), padding=1)
    self.decoder_entry = nn.Sequential(
        decoder_entry_conv,
        nn.PixelShuffle(2)
      )

    nn.init.normal_(decoder_entry_conv.weight, mean=1/100., std=1/10.)

    self.decoder_residuals = nn.ModuleList([
      Residual(),
      Residual(),
      Residual()
    ])
    # Following three residual blocks,
    #   two sub-pixel convolution layers upsample the image to the resolution of the input.
    self.decoder_exit = nn.Sequential(
        nn.Conv2d(128, 256, (3, 3), padding=1),
        nn.PixelShuffle(2),
        nn.Conv2d(64, 12, (3, 3), padding=1),
        nn.PixelShuffle(2)
      )

  def forward(self, x):
    # The first two layers of the encoder perform preprocessing,
    #   namely mirror padding and a fixed pixelwise normalization.
    # The mirror-padding was chosen such that the output of the encoder has the same spatial extent as an 8 times downsampled image.
    # The normalization centers the distribution of each channel’s values and ensures it has approximately unit variance.
    x = F.pad(x, (14, 14, 14, 14), mode='reflect')

    x = (x.clone() - means_tensor[None, :, None, None]) / std_tensor[None, :, None, None]

    x = self.encoder_entry(x)
    for r in self.encoder_residuals:
      x = r(x)
    x = self.encoder_exit(x)

    x = RoundIdGradient.apply(x)
    code = x

    x = self.decoder_entry(x)
    for r in self.decoder_residuals:
      x = r(x)
    x = self.decoder_exit(x)

    x = x.clone() * std_tensor[None, :, None, None] + means_tensor[None, :, None, None]

    # Finally, after denormalization, the pixel values are clipped to the range of 0 to 255.
    # Similar to how we deal with gradients of the rounding function,
    #   we redefine the gradient of the clipping function to be 1 outside the clipped range.
    # This ensures that the training signal is non-zero even when the decoded pixels are outside this range (Appendix A.1).
    x = ByteClampIdGradient.apply(x)
    return code, x

import pickle
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Normalize

from device import device
from autograd.ByteClampIdGradient import ByteClampIdGradient
from autograd.RoundIdGradient import RoundIdGradient

train_stats = pickle.load(open('data/train_stats.pickle', 'rb'))

means_tensor = torch.tensor([train_stats['means']['r'], train_stats['means']['g'], train_stats['means']['b']]).to(device)
std_tensor = torch.tensor([train_stats['std']['r'], train_stats['std']['g'], train_stats['std']['b']]).to(device)

# Thanks catta202000! [https://github.com/pytorch/pytorch/pull/5429/files]
def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
    """Fills the input Tensor or Variable with values according to the method
    described in "Checkerboard artifact free sub-pixel convolution"
    - Andrew Aitken et al. (2017), this inizialization should be used in the
    last convolutional layer before a PixelShuffle operation
    Args:
        tensor: an n-dimensional torch.Tensor or autograd.Variable
        upscale_factor: factor to increase spatial resolution by
        inizializer: inizializer to be used for sub_kernel inizialization
    Examples:
        >>> upscale = 8
        >>> num_classes = 10
        >>> previous_layer_features = Variable(torch.Tensor(8, 64, 32, 32))
        >>> conv_shuffle = Conv2d(64, num_classes * (upscale ** 2), 3, padding=1, bias=0)
        >>> ps = PixelShuffle(upscale)
        >>> kernel = ICNR(conv_shuffle.weight, scale_factor=upscale)
        >>> conv_shuffle.weight.data.copy_(kernel)
        >>> output = ps(conv_shuffle(previous_layer_features))
        >>> print(output.shape)
        torch.Size([8, 10, 256, 256])
    .. _Checkerboard artifact free sub-pixel convolution:
        https://arxiv.org/abs/1707.02937
    """
    new_shape = [int(tensor.shape[0] / (upscale_factor ** 2))] + list(tensor.shape[1:])
    subkernel = torch.zeros(new_shape)
    inizializer(subkernel)
    subkernel = subkernel.transpose(0, 1)

    subkernel = subkernel.contiguous().view(subkernel.shape[0],
                                            subkernel.shape[1], -1)

    kernel = subkernel.repeat(1, 1, upscale_factor ** 2)

    transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor.shape[2:])
    kernel = kernel.contiguous().view(transposed_shape)

    kernel = kernel.transpose(0, 1)

    return kernel

def conv_init(t):
  nn.init.kaiming_normal_(t)

  # # Convolution Aware Initialization [https://arxiv.org/pdf/1702.06295.pdf]

  # print(c.weight.shape)

  # # this part of the source from pytorch nn.init.orthogonal_ START
  # rows = c.weight.size(0)
  # cols = c.weight.numel() // rows
  # noise = c.weight.new(rows, cols).normal_(0, 1)

  # if rows < cols:
  #     noise.t_()
  # print(noise.shape)

  # noise = noise.rfft(2)
  # print(noise.shape)

  # # Compute the qr factorization
  # q, r = torch.qr(noise)
  # # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
  # d = torch.diag(r, 0)
  # ph = d.sign()
  # q *= ph

  # if rows < cols:
  #     q.t_()

  # # nn.init.orthogonal_ OVER

  # c.weight = q
  # c.weight = c.weight.irfft(2)

  # fin, fout = nn.init._calculate_fan_in_and_fan_out(c.weight)
  # c.weight *= torch.sqrt(2./fin / c.weight.var())
  # # todo: is this right????

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

    for c in self.convs:
      conv_init(c.weight)

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

    convs = []

    # Afterwards, the image is convolved and spatially downsampled while at the same time increasing the number of channels to 128.
    en_entry_convs = [
      nn.Conv2d(3, 64, (5, 5), stride=2),
      nn.Conv2d(64, 128, (5, 5), stride=2)
    ]
    convs.extend(en_entry_convs)
    self.encoder_entry = nn.Sequential(
        en_entry_convs[0],
        nn.ReLU(),
        nn.BatchNorm2d(64),

        en_entry_convs[1],
        nn.ReLU(),
        nn.BatchNorm2d(128)
      )

    # This is followed by three residual blocks (He et al., 2015),
    #   where each block consists of an additional two convolutional layers with 128 filters each.
    self.encoder_residuals = nn.ModuleList([
      Residual(),
      Residual(),
      Residual(),
      nn.ReLU(),
      nn.BatchNorm2d(128)
    ])
    # A final convolutional layer is applied and the coefficients downsampled again before quantization through rounding to the nearest integer.
    self.encoder_exit = nn.Conv2d(128, 96, (5, 5), stride=2)
    convs.append(self.encoder_exit)

    # The decoder mirrors the architecture of the encoder (Figure 9).
    #   Instead of mirror-padding and valid convolutions, we use zero-padded convolutions.

    # Upsampling is achieved through convolution followed by a reorganization of the coefficients.
    #   This reorganization turns a tensor with many channels into a tensor of the same dimensionality but with fewer channels and larger spatial extent (for details, see Shi et al., 2016).
    # A convolution and reorganization of coefficients together form a sub-pixel convolution layer.
    decoder_entry_conv = nn.Conv2d(96, 512, (3, 3), padding=1)
    convs.append(decoder_entry_conv)
    self.decoder_entry = nn.Sequential(
        decoder_entry_conv,
        nn.PixelShuffle(2)
      )

    self.decoder_residuals = nn.ModuleList([
      Residual(),
      Residual(),
      Residual(),
      nn.ReLU(),
      nn.BatchNorm2d(128)
    ])
    # Following three residual blocks,
    #   two sub-pixel convolution layers upsample the image to the resolution of the input.
    decoder_exit_convs = [
      nn.Conv2d(128, 256, (3, 3), padding=1),
      nn.Conv2d(64, 12, (3, 3), padding=1)
    ]
    self.decoder_exit = nn.Sequential(
        decoder_exit_convs[0],
        nn.PixelShuffle(2),
        nn.ReLU(),
        nn.BatchNorm2d(64),

        decoder_exit_convs[1],
        nn.PixelShuffle(2)
      )
    for c in decoder_exit_convs:
      w = ICNR(c.weight, upscale_factor=2, inizializer=conv_init)
      c.weight.data.copy_(w)

    for c in convs:
      conv_init(c.weight)

  def forward(self, x):
    code = self.encode(x)
    x = self.decode(code.clone())
    return code, x

  def encode(self, x):
    # debug = []

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
    x = ByteClampIdGradient.apply(x)
    code = x
    return code

  def decode(self, code):
    x = code
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
    # return debug, code, x
    return x

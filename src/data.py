import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# All models were trained using Adam (Kingma & Ba, 2015) applied to batches of 32 images 128×128 pixels in size.
def load_dataset(data_path):
  return ImageFolder(
    root=data_path,
    transform=transforms.Compose([
        # transforms.RandomAffine(30, shear=(-5, 5, -5, 5)), # don't want empty areas
        transforms.RandomResizedCrop(128, scale=(.25, .6)),
        transforms.ColorJitter(brightness=.1, contrast=.05, saturation=.05, hue=.05),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(.05),
        transforms.ToTensor()
      ])
  )

train_loader = DataLoader(
  load_dataset('data/train'),
  batch_size=32,
  num_workers=4,
  shuffle=True,
  pin_memory=True
)
eval_loader = DataLoader(
  load_dataset('data/train'),
  batch_size=10,
  num_workers=4,
  shuffle=True
)

import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# All models were trained using Adam (Kingma & Ba, 2015) applied to batches of 32 images 128×128 pixels in size.
def load_dataset():
  data_path = 'data/images/Images'

  train_dataset = ImageFolder(
    root=data_path,
    transform=transforms.Compose([
        transforms.RandomCrop(128),
        transforms.ToTensor()
      ])
  )
  train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
    pin_memory=True
  )
  return train_loader
train_loader = load_dataset()

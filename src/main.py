import os
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Net
from cli import args
from data import train_loader
from visuals import show_side_by_side

use_cuda = torch.cuda.is_available() and not args.no_cuda
print(f'CUDA found={torch.cuda.is_available()}')
print(f'CUDA={use_cuda}, N GPUs={torch.cuda.device_count()}')
device = torch.device('cuda' if use_cuda else 'cpu')

# determinism is good
torch.manual_seed(0)
if use_cuda:
  torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = Net()
if torch.cuda.device_count() > 1:
  model = nn.DataParallel(model)
model = model.to(device)

optimiser = optim.Adam(model.parameters())

model.train()
train_losses = []

if args.resume:
  print('Loading from checkpoint')
  model.load_state_dict(torch.load('checkpoint/model.pth'))
  optimiser.load_state_dict(torch.load('checkpoint/optimiser.pth'))

# fixme: epochs aren't epochs
history = []
for epoch in range(args.epochs):
  for i, (data, target) in enumerate(train_loader):
    batch_start = time.time()

    data = data.to(device=device, non_blocking=True)
    target = data

    optimiser.zero_grad()

    code, output = model(data)

    loss = F.mse_loss(output, target)
    loss.backward()
    train_losses.append(loss.item())

    optimiser.step()

    batch_end = time.time()

    if i % args.save_interval == 0:
      history.append((target[0], output[0].round().detach()))
      history = history[-10:]
      show_side_by_side(history)

      print(f'  Batch {i}: loss={loss.item():.2} ~ {1/(batch_end-batch_start):.2} b/s')

      torch.save(model.state_dict(), 'checkpoint/model1.pth')
      os.replace('checkpoint/model1.pth', 'checkpoint/model.pth')
      torch.save(optimiser.state_dict(), 'checkpoint/optimiser1.pth')
      os.replace('checkpoint/optimiser1.pth', 'checkpoint/optimiser.pth')
  print(f'Epoch {epoch}: avg loss={mean(train_losses):.2}')
  train_losses = []

print('Done!')

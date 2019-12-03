import pickle
import os
import time
from statistics import mean

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Net
from cli import args
from data import train_loader
from visuals import show_side_by_side

# show_side_by_side([1, 2, 3])

# import sys
# sys.exit(0)

from device import device, use_cuda

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

optimiser = optim.Adam(model.parameters(), amsgrad=True)

model.train()
train_losses = []

chk_data = {
  'lastBatch': 0,
  'lastEpoch': 0
}
if args.resume:
  print('Loading from checkpoint')
  model.load_state_dict(torch.load('checkpoint/model.pth'))
  optimiser.load_state_dict(torch.load('checkpoint/optimiser.pth'))

  chk_data = pickle.load(open('checkpoint/data.pickle', 'rb'))

  print(f'Starting from {chk_data["lastEpoch"]}/{chk_data["lastBatch"]}')

# fixme: epochs aren't epochs
history = []
for epoch in range(args.epochs):
  torch.manual_seed(chk_data['lastEpoch'])

  load_iter = iter(train_loader)
  for j in range(chk_data['lastBatch']):
    load_iter.__next__()

  for i, (data, target) in enumerate(load_iter):
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

    if chk_data["lastBatch"] % args.save_interval == 0:
      normedCode = (code[0] + (-code[0]).max()) / (code[0].max() - code[0].min())

      history.append((target[0], output[0].round().detach(), normedCode.detach()))
      history = history[-10:]
      show_side_by_side(history)

      print(f'  Batch {chk_data["lastEpoch"]}/{chk_data["lastBatch"]}: loss={loss.item():.2} ~ {1/(batch_end-batch_start):.2} b/s')
      print(f'    Code mean={code[0].mean():.5} std={code[0].std():.5}')

      chk_data['lastBatch'] += 1
      pickle.dump(chk_data, open('checkpoint/data1.pickle', 'wb'))
      os.replace('checkpoint/data1.pickle', 'checkpoint/data.pickle')

      torch.save(model.state_dict(), 'checkpoint/model1.pth')
      os.replace('checkpoint/model1.pth', 'checkpoint/model.pth')
      torch.save(optimiser.state_dict(), 'checkpoint/optimiser1.pth')
      os.replace('checkpoint/optimiser1.pth', 'checkpoint/optimiser.pth')
  print(f'Epoch {chk_data["lastEpoch"]}: avg loss={mean(train_losses):.2}')
  train_losses = []

  chk_data['lastEpoch'] += 1
  chk_data['lastBatch'] = 0

  pickle.dump(chk_data, open('checkpoint/data1.pickle', 'wb'))
  os.replace('checkpoint/data1.pickle', 'checkpoint/data.pickle')

print('Done!')

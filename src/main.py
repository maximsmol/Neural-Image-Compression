import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Net
from cli import args
from data import train_loader
from visuals import show_side_by_side

# determinism is good
use_cuda = torch.cuda.is_available() and not args.no_cuda
device = torch.device('cuda' if use_cuda else 'cpu')
torch.manual_seed(0)
if use_cuda:
  torch.cuda.manual_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model = Net().to(device)
optimiser = optim.Adam(model.parameters())

model.train()
train_losses = []

if args.resume:
  model.load_state_dict(torch.load('model.pth'))
  optimiser.load_state_dict(torch.load('optimiser.pth'))

# fixme: epochs aren't epochs
for i, (data, target) in enumerate(train_loader):
  if i >= args.epochs:
    break

  data = data.to(device=device, non_blocking=True)
  target = data

  optimiser.zero_grad()

  code, output = model(data)

  loss = F.mse_loss(output, target)
  loss.backward()
  train_losses.append(loss.item())

  optimiser.step()

  show_side_by_side(target[0], output[0].round().detach())

  if i % args.save_interval == 0 or i == args.epochs - 1:
    print(f'Batch {i}: loss={loss.item()}')
    torch.save(model.state_dict(), 'checkpoint/model.pth')
    torch.save(optimiser.state_dict(), 'checkpoint/optimiser.pth')
    torch.save(train_losses, 'checkpoint/train_losses.pth')

print('Done!')

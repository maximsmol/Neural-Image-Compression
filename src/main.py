import pickle
import os
import time
import sys
from statistics import mean
from os.path import join
from shutil import copyfile, rmtree
from math import log

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import Net
from cli import args
from data import train_loader, eval_loader
import visuals

from device import device
import determinism

print('Creating a model instance')

model = Net()
if torch.cuda.device_count() > 1:
  print('  Setting up data parallelilsm')
  model = nn.DataParallel(model)
model = model.to(device)

checkpoint_root = 'checkpoint'
latest_checkpoint_dir = join(checkpoint_root, 'latest')

have_checkpoint = os.path.exists(join(latest_checkpoint_dir, 'model.pth'))
should_resume = not args.restart and have_checkpoint

if should_resume:
  print('  Loading the model from the latest checkpoint')
  model.load_state_dict(torch.load(join(latest_checkpoint_dir, 'model.pth')))
else:
  if have_checkpoint:
    rmtree(checkpoint_root)
  if os.path.exists('log'):
    rmtree('log')

os.makedirs(latest_checkpoint_dir, exist_ok=True)

if args.mode == 'train':
  print('Setting up training')

  optimiser = optim.Adam(model.parameters(), amsgrad=True)
  model.train()


  chk_data = {
    'lastBatch': 0,
    'lastEpoch': 0,
    'lastBatchId': 0,
    'checkpointN': 0,

    'trainLosses': [],
    'evalLosses': []
  }
  if should_resume:
    print('  Loading training state from the latest checkpoint')

    optimiser.load_state_dict(torch.load(join(latest_checkpoint_dir, 'optimizer.pth')))
    chk_data = pickle.load(open(join(latest_checkpoint_dir, 'data.pickle'), 'rb'))

    print(f'    Continuing from checkpoint {chk_data["checkpointN"]+1}, batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]+1}')

  writer = SummaryWriter('log', purge_step=chk_data['lastBatchId'])
  writer.add_custom_scalars({
    'Log-loss': {
      'Batch': [
        'Multiline',
        ['Log-loss/train', 'Log-loss/eval']
      ],
      'Epoch average': [
        'Multiline',
        ['Log-loss-epoch-avg/train', 'Log-loss-epoch-avg/eval']
      ]
    },
    'Loss': {
      'Batch': [
        'Multiline',
        ['Loss/train', 'Loss/eval']
      ],
      'Epoch average': [
        'Multiline',
        ['Loss-epoch-avg/train', 'Loss-epoch-avg/eval']
      ]
    }
  })

  cur_time = time.time()

  log_time = cur_time
  checkpoint_time = cur_time
  backup_time = cur_time

  print('Training')
  eval_iter = iter(eval_loader)
  while True:
    epoch_start = time.time()

    torch.manual_seed(epoch_start)
    load_iter = iter(train_loader)

    batches_processed = 0

    def write_checkpoint():
      print(f'  Writing checkpoint {chk_data["checkpointN"]+1} for batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]}')

      nonlocal checkpoint_time
      checkpoint_time = cur_time


      open(join(latest_checkpoint_dir, 'lock'), 'w')


      pickle.dump(chk_data, open(join(latest_checkpoint_dir, 'data1.pickle'), 'wb')) # todo: code reuse
      os.replace(join(latest_checkpoint_dir, 'data1.pickle'), join(latest_checkpoint_dir, 'data.pickle'))


      torch.save(model.state_dict(), join(latest_checkpoint_dir, 'model1.pth'))
      os.replace(join(latest_checkpoint_dir, 'model1.pth'), join(latest_checkpoint_dir, 'model.pth'))

      torch.save(optimiser.state_dict(), join(latest_checkpoint_dir, 'optimizer1.pth'))
      os.replace(join(latest_checkpoint_dir, 'optimizer1.pth'), join(latest_checkpoint_dir, 'optimizer.pth'))


      os.remove(join(latest_checkpoint_dir, 'lock'))

      nonlocal backup_time
      if cur_time - backup_time >= args.backup_interval:
        print(f'    Saving a backup')

        backup_time = cur_time

        checkpoint_dir = join(checkpoint_root, str(chk_data["checkpointN"]+1))
        os.makedirs(checkpoint_dir, exist_ok=True)

        open(join(checkpoint_dir, 'lock'), 'w')

        for f in os.listdir(latest_checkpoint_dir):
          copyfile(join(latest_checkpoint_dir, f), join(checkpoint_dir, f))

        os.remove(join(checkpoint_dir, 'lock'))

      chk_data["checkpointN"] += 1
      print('    Done')

    if chk_data['lastEpoch'] >= args.epochs:
      print('Reached epoch limit')
      write_checkpoint()
      break

    for i, (data, _) in enumerate(load_iter):
      data = data.to(device=device, non_blocking=True)
      target = data


      optimiser.zero_grad()

      # debug, code, output = model(data)
      code, output = model(data)

      loss = F.l1_loss(output, target)
      loss.backward()

      optimiser.step()

      batches_processed += 1
      cur_time = time.time()


      writer.add_scalar('Log-loss/train', log(loss.item()), global_step=chk_data['lastBatchId'], walltime=cur_time)
      writer.add_scalar('Loss/train', loss.item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
      chk_data['trainLosses'].append(loss.item())

      if cur_time - log_time >= args.log_interval or args.debug_single_batch:
        log_time = cur_time


        model.eval()

        try:
          eval_data, _ = eval_iter.__next__()
        except StopIteration:
          eval_iter = iter(eval_loader)
          eval_data, _ = eval_iter.__next__()

        eval_data = eval_data.to(device=device, non_blocking=True)
        # _, eval_code, eval_output = model(eval_data)
        eval_code, eval_output = model(eval_data)
        eval_loss = F.l1_loss(eval_output, eval_data)

        writer.add_scalar('Log-loss/eval', log(eval_loss.item()), global_step=chk_data['lastBatchId'], walltime=cur_time)
        writer.add_scalar('Loss/eval', loss.item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
        chk_data['evalLosses'].append(eval_loss.item())

        model.train()


        normedCode = (code[0] + (-code[0]).max()) / (code[0].max() - code[0].min())

        imgs = (target[0].cpu(), output[0].detach().cpu(), normedCode.detach().cpu())
        writer.add_figure('Side-by-side', visuals.show_side_by_side(imgs), global_step=chk_data['lastBatchId'], walltime=cur_time)

        print(f'  Batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]+1}: train loss={loss.item():.2} eval loss={eval_loss.item():.2} ~ {batches_processed/(cur_time-epoch_start):.2} b/s')
        # for i in range(len(debug)):
        #   print(f'    Debug={debug[i].cpu()}')
        #   for c in range(debug[i].size(1)):
        #     writer.add_histogram(f'Debug/{i}-{c}', debug[i][0][c], global_step=chk_data['lastBatchId'], walltime=cur_time)
        #   print(f'    Debug {i} mean={debug[i].mean():.5} std={debug[i].std():.5}')

        writer.flush()

      chk_data['lastBatch'] += 1
      chk_data['lastBatchId'] += 1

      if cur_time - checkpoint_time >= args.save_interval:
        write_checkpoint()
      if args.debug_single_batch:
        break;

    trainl = mean(chk_data["trainLosses"])
    writer.add_scalar('Log-loss-epoch-avg/train', log(trainl), global_step=chk_data['lastBatchId'], walltime=cur_time)
    writer.add_scalar('Loss-epoch-avg/train', trainl, global_step=chk_data['lastBatchId'], walltime=cur_time)
    if not chk_data["evalLosses"]:
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={trainl:.2} avg eval loss=n/a') # todo: code reuse
    else:
      evall = mean(chk_data["evalLosses"])
      writer.add_scalar('Log-loss-epoch-avg/eval', log(evall), global_step=chk_data['lastBatchId'], walltime=cur_time)
      writer.add_scalar('Loss-epoch-avg/eval', evall, global_step=chk_data['lastBatchId'], walltime=cur_time)
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={trainl:.2} avg eval loss={evall:.2}')

    chk_data['evalLosses'] = []
    chk_data['trainLosses'] = []
    chk_data['lastEpoch'] += 1
    chk_data['lastBatch'] = 0

    writer.flush()

    if args.debug_single_batch:
      break;

writer.flush()
writer.close()
print('Done!')

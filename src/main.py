import pickle
import os
import time
import sys
from statistics import mean
from os.path import join
from shutil import copyfile

import torch
import torch.optim as optim
import torch.nn.functional as F

from model import Net
from cli import args
from data import train_loader, eval_loader
import visuals


if args.mode == 'render-log':
  print('UNSUPPORTED for now')
  sys.exit(0)

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

if args.resume:
  print('  Loading the model from the latest checkpoint')

  model.load_state_dict(torch.load(join(latest_checkpoint_dir, 'model.pth')))
else:
  if os.path.exists(latest_checkpoint_dir):
    print(f'Found a saved checkpoint. Resume with --resume or clear {checkpoint_root}/* to restart')
    print('Quitting')
    sys.exit(0)
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
    'trainLosses': [[]],
    'evalLosses': [[]],
    'imghistory': []
  }
  if args.resume:
    print('  Loading training state from the latest checkpoint')

    optimiser.load_state_dict(torch.load(join(latest_checkpoint_dir, 'optimizer.pth')))
    chk_data = pickle.load(open(join(latest_checkpoint_dir, 'data.pickle'), 'rb'))

    print(f'    Continuing from checkpoint {chk_data["checkpointN"]+1}, batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]+1}')

  cur_time = time.time()

  log_time = cur_time
  checkpoint_time = cur_time
  backup_time = cur_time

  print('Training')
  while True:
    if chk_data['lastEpoch'] >= args.epochs:
      print('Reached epoch limit')
      break

    epoch_start = time.time()

    torch.manual_seed(epoch_start)
    load_iter = iter(train_loader)
    eval_iter = iter(eval_loader)

    batches_processed = 0
    for i, (data, _) in enumerate(load_iter):
      data = data.to(device=device, non_blocking=True)
      target = data


      optimiser.zero_grad()

      code, output = model(data)

      loss = F.l1_loss(output, target)
      loss.backward()
      chk_data['trainLosses'][-1].append(loss.item())

      optimiser.step()


      batches_processed += 1
      cur_time = time.time()
      if cur_time - log_time >= args.log_interval:
        log_time = cur_time


        model.eval()

        eval_data, _ = eval_iter.__next__()
        eval_data = eval_data.to(device=device, non_blocking=True)
        eval_code, eval_output = model(eval_data)
        eval_loss = F.l1_loss(eval_output, eval_data)
        chk_data['evalLosses'][-1].append({
          'batch': chk_data["lastBatch"],
          'batchId': chk_data["lastBatchId"],
          'loss': eval_loss.item()
        })

        model.train()


        code_min = code[0].min()
        code_max = code[0].max()
        code_range = code_max - code_min

        normedCode = (code[0] + (-code[0]).max()) / code_range

        os.makedirs('log/', exist_ok=True)

        chk_data['imghistory'].append((target[0].cpu(), output[0].round().detach().cpu(), normedCode.detach().cpu()))
        chk_data['imghistory'] = chk_data['imghistory'][-10:]
        visuals.show_side_by_side(chk_data['imghistory']).savefig('log/side-by-side.pdf')

        visuals.show_loss_graphs(chk_data['trainLosses']).savefig('log/train-loss.pdf')
        visuals.show_loss_graphs(chk_data['evalLosses'], eval=True).savefig('log/eval-loss.pdf')

        print(f'  Batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]+1}: train loss={loss.item():.2} eval loss={eval_loss.item():.2} ~ {batches_processed/(cur_time-epoch_start):.2} b/s')
        print(f'    Code mean={code[0].mean():.5} std={code[0].std():.5} min={code[0].min():.5} min={code[0].max():.5} range={code_range:.5}')


      chk_data['lastBatch'] += 1
      chk_data['lastBatchId'] += 1

      if cur_time - checkpoint_time >= args.save_interval:
        print(f'  Writing checkpoint {chk_data["checkpointN"]+1} for batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]}')

        checkpoint_time = cur_time


        open(join(latest_checkpoint_dir, 'lock'), 'w')


        pickle.dump(chk_data, open(join(latest_checkpoint_dir, 'data1.pickle'), 'wb')) # todo: code reuse
        os.replace(join(latest_checkpoint_dir, 'data1.pickle'), join(latest_checkpoint_dir, 'data.pickle'))


        torch.save(model.state_dict(), join(latest_checkpoint_dir, 'model1.pth'))
        os.replace(join(latest_checkpoint_dir, 'model1.pth'), join(latest_checkpoint_dir, 'model.pth'))

        torch.save(optimiser.state_dict(), join(latest_checkpoint_dir, 'optimizer1.pth'))
        os.replace(join(latest_checkpoint_dir, 'optimizer1.pth'), join(latest_checkpoint_dir, 'optimizer.pth'))


        os.remove(join(latest_checkpoint_dir, 'lock'))

        if cur_time - backup_time >= args.backup_interval:
          print(f'    Saving a backup')

          checkpoint_dir = join(checkpoint_root, str(chk_data["checkpointN"]+1))
          os.makedirs(checkpoint_dir, exist_ok=True)

          open(join(checkpoint_dir, 'lock'), 'w')

          for f in os.listdir(latest_checkpoint_dir):
            copyfile(join(latest_checkpoint_dir, f), join(checkpoint_dir, f))

          os.remove(join(checkpoint_dir, 'lock'))

        chk_data["checkpointN"] += 1
        print('    Done')

    if not chk_data["evalLosses"][-1]:
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={mean(chk_data["trainLosses"][-1]):.2} avg eval loss=n/a') # todo: code reuse
    else:
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={mean(chk_data["trainLosses"][-1]):.2} avg eval loss={mean([x["loss"] for x in chk_data["evalLosses"][-1]]):.2}')

    chk_data['evalLosses'].append([])
    chk_data['trainLosses'].append([])
    chk_data['lastEpoch'] += 1
    chk_data['lastBatch'] = 0

    pickle.dump(chk_data, open('checkpoint/latest/data1.pickle', 'wb')) # todo: code reuse
    os.replace('checkpoint/latest/data1.pickle', 'checkpoint/latest/data.pickle')

print('Done!')

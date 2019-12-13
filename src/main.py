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
from torchvision.transforms import ToPILImage, ToTensor

import numpy as np
import PIL

from pytorch_msssim import SSIM

from model import Net
from prob_model import ProbabilityModel
from cli import args
from data import train_loader, eval_loader
import visuals

from device import device
import determinism
from cielab_transform import InvCIELABTransform

# import arithmeticcoding

print('Creating a model instance')

model = Net()
probability_model = ProbabilityModel()
if torch.cuda.device_count() > 1:
  print('  Setting up data parallelilsm')
  model = nn.DataParallel(model)
  probability_model = nn.DataParallel(probability_model)
model = model.to(device)
probability_model = probability_model.to(device)

checkpoint_root = 'checkpoint'
latest_checkpoint_dir = join(checkpoint_root, 'latest')

have_checkpoint = os.path.exists(join(latest_checkpoint_dir, 'model.pth'))
should_resume = not args.restart and have_checkpoint

if should_resume:
  print('  Loading the model from the latest checkpoint')
  model.load_state_dict(torch.load(join(latest_checkpoint_dir, 'model.pth')))
  probability_model.load_state_dict(torch.load(join(latest_checkpoint_dir, 'probability_model.pth')))
else:
  if have_checkpoint:
    rmtree(checkpoint_root)
  if os.path.exists('log'):
    rmtree('log')

os.makedirs(latest_checkpoint_dir, exist_ok=True)

if args.mode == 'train':
  print('Setting up training')

  optimiser = optim.Adam(
    list(model.parameters()) + list(probability_model.parameters()),
    amsgrad=True
  )
  model.train()
  probability_model.train()

  chk_data = {
    'lastBatch': 0,
    'lastEpoch': 0,
    'lastBatchId': 0,
    'checkpointN': 0,

    'trainLosses': [],
    'evalLosses': [],
    'trainLikelihoods': [],
    'evalLikelihoods': []
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
    },
    'Likelihood': {
      'Batch': [
        'Multiline',
        ['Likelihood/train', 'Likelihood/eval']
      ],
      'Epoch average': [
        'Multiline',
        ['Likelihood-epoch-avg/train', 'Likelihood-epoch-avg/eval']
      ]
    }
  })

  to_tensor = ToTensor()
  def to_pilimagelab(pic):
    pic = pic.mul(255).byte()
    nppic = np.transpose(pic.numpy(), (1, 2, 0))
    return PIL.Image.fromarray(nppic, mode='LAB')
  invcielab = InvCIELABTransform()

  ssim = SSIM(win_size=11, win_sigma=1.5, data_range=255, size_average=True, channel=3)

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

      global checkpoint_time
      checkpoint_time = cur_time


      open(join(latest_checkpoint_dir, 'lock'), 'w')


      pickle.dump(chk_data, open(join(latest_checkpoint_dir, 'data1.pickle'), 'wb')) # todo: code reuse
      os.replace(join(latest_checkpoint_dir, 'data1.pickle'), join(latest_checkpoint_dir, 'data.pickle'))


      torch.save(model.state_dict(), join(latest_checkpoint_dir, 'model1.pth'))
      os.replace(join(latest_checkpoint_dir, 'model1.pth'), join(latest_checkpoint_dir, 'model.pth'))

      torch.save(optimiser.state_dict(), join(latest_checkpoint_dir, 'probability_model1.pth'))
      os.replace(join(latest_checkpoint_dir, 'probability_model1.pth'), join(latest_checkpoint_dir, 'probability_model.pth'))

      torch.save(optimiser.state_dict(), join(latest_checkpoint_dir, 'optimizer1.pth'))
      os.replace(join(latest_checkpoint_dir, 'optimizer1.pth'), join(latest_checkpoint_dir, 'optimizer.pth'))


      os.remove(join(latest_checkpoint_dir, 'lock'))

      global backup_time
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

    for i, (data_cie, _) in enumerate(load_iter):
      data_cie = data_cie.to(device=device, non_blocking=True)
      target = data_cie


      optimiser.zero_grad()

      # debug, code, output = model(data_cie)
      code, output = model(data_cie)
      likelihood = probability_model.likelihood(code)/(96*128*128/64)

      distortion_loss = 1-ssim(output, target)
      loss = likelihood + distortion_loss
      loss.backward()

      optimiser.step()

      batches_processed += 1
      cur_time = time.time()


      writer.add_scalar('Distortion/train', (distortion_loss / 32).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
      writer.add_scalar('Likelihood/train', (likelihood / 32).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)

      writer.add_scalar('Log-loss/train', log((loss / 32).item()), global_step=chk_data['lastBatchId'], walltime=cur_time)
      writer.add_scalar('Loss/train', (loss / 32).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
      chk_data['trainLosses'].append((loss / 32).item())
      chk_data['trainLikelihoods'].append((likelihood / 32).item())

      if cur_time - log_time >= args.log_interval or args.debug_single_batch:
        log_time = cur_time


        model.eval()

        try:
          eval_data_cie, _ = eval_iter.__next__()
        except StopIteration:
          eval_iter = iter(eval_loader)
          eval_data_cie, _ = eval_iter.__next__()

        eval_data_cie = eval_data_cie.to(device=device, non_blocking=True)

        # _, eval_code, eval_output = model(eval_data_cie)
        eval_code, eval_output = model(eval_data_cie)
        eval_likelihood = probability_model.likelihood(eval_code)/(96*128*128/64)

        eval_distortion_loss = 1-ssim(eval_output, eval_data_cie)
        eval_loss = eval_likelihood + eval_distortion_loss

        writer.add_scalar('Likelihood/eval', (eval_likelihood / 10).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
        writer.add_scalar('Distortion/eval', (eval_distortion_loss / 10).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)

        writer.add_scalar('Log-loss/eval', log((eval_loss / 10).item()), global_step=chk_data['lastBatchId'], walltime=cur_time)
        writer.add_scalar('Loss/eval', (loss / 10).item(), global_step=chk_data['lastBatchId'], walltime=cur_time)
        chk_data['evalLosses'].append((eval_loss / 10).item())
        chk_data['evalLikelihoods'].append((likelihood / 10).item())

        most_probable = torch.where(
          probability_model.probs > .5,
          torch.ones_like(eval_code[0]),
          torch.zeros_like(eval_code[0])
        )
        most_probable_decoded = model.decode(most_probable)

        probability_model_sample = torch.rand_like(eval_code[0])
        probability_model_sample = torch.where(
          probability_model_sample > probability_model.probs,
          torch.ones_like(probability_model_sample),
          torch.zeros_like(probability_model_sample)
        )
        probability_model_sample_decoded = model.decode(probability_model_sample)

        writer.add_histogram('Probability model', probability_model.probs, global_step=chk_data['lastBatchId'], walltime=cur_time)
        writer.add_image('Probability-model-sample', probability_model_sample_decoded[0], global_step=chk_data['lastBatchId'], walltime=cur_time)
        writer.add_image('Most probable sample', most_probable_decoded[0], global_step=chk_data['lastBatchId'], walltime=cur_time)

        model.train()


        normedCode = (code[0] + (-code[0]).max()) / (code[0].max() - code[0].min())

        data_rgb = data_cie[0].cpu()
        output_rgb = output[0].detach().cpu()
        # data_rgb = to_tensor(invcielab(to_pilimagelab(data_cie[0].cpu())))
        # output_rgb = to_tensor(invcielab(to_pilimagelab(output[0].detach().cpu())))

        imgs = (data_rgb, output_rgb, normedCode.detach().cpu())
        writer.add_figure('Side-by-side', visuals.show_side_by_side(imgs), global_step=chk_data['lastBatchId'], walltime=cur_time)

        print(f'  Batch {chk_data["lastEpoch"]+1}/{chk_data["lastBatch"]+1}: train loss={(loss / 32).item():.2} eval loss={(eval_loss / 10).item():.2} ~ {batches_processed/(cur_time-epoch_start):.2} b/s')
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

    trainlikelihood = mean(chk_data["trainLikelihoods"])
    trainl = mean(chk_data["trainLosses"])

    writer.add_scalar('Likelihood-epoch-avg/train', trainlikelihood, global_step=chk_data['lastBatchId'], walltime=cur_time)

    writer.add_scalar('Log-loss-epoch-avg/train', log(trainl), global_step=chk_data['lastBatchId'], walltime=cur_time)
    writer.add_scalar('Loss-epoch-avg/train', trainl, global_step=chk_data['lastBatchId'], walltime=cur_time)
    if not chk_data["evalLosses"]:
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={trainl:.2} avg eval loss=n/a') # todo: code reuse
    else:
      evallikelihood = mean(chk_data["evalLikelihoods"])
      evall = mean(chk_data["evalLosses"])

      writer.add_scalar('Likelihood-epoch-avg/eval', evallikelihood, global_step=chk_data['lastBatchId'], walltime=cur_time)

      writer.add_scalar('Log-loss-epoch-avg/eval', log(evall), global_step=chk_data['lastBatchId'], walltime=cur_time)
      writer.add_scalar('Loss-epoch-avg/eval', evall, global_step=chk_data['lastBatchId'], walltime=cur_time)
      print(f'Epoch {chk_data["lastEpoch"]+1}: avg train loss={trainl:.2} avg eval loss={evall:.2}')

    chk_data['trainLikelihoods'] = []
    chk_data['evalLikelihoods'] = []

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

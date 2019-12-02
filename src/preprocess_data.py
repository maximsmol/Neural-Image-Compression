import pickle
import os
import sys
import random
import math
from os.path import exists, join

from PIL import Image
import numpy as np

data_dir = 'data'

raw_imgs = join(data_dir, 'images', 'Images')

train_imgs = join(data_dir, 'train', 'dummy_class')
eval_imgs = join(data_dir, 'eval', 'dummy_class')
test_imgs = join(data_dir, 'test', 'dummy_class')

if exists(train_imgs):
  print('Data already split!')
  print('Will only recompute train set statistics')
else:
  os.makedirs(train_imgs, exist_ok=True)
  os.makedirs(eval_imgs, exist_ok=True)
  os.makedirs(test_imgs, exist_ok=True)

  random.seed(0)

  eval_percent = 10
  test_percent = 10
  train_percent = 100 - eval_percent - test_percent


  for breed in os.listdir(raw_imgs):
    breed_dir = join(raw_imgs, breed)
    for img in os.listdir(breed_dir):
      path = join(breed_dir, img)

      x = random.uniform(0, 100)
      dir = ''
      if x < train_percent:
        dir = train_imgs
      elif x < train_percent + eval_percent:
        dir = eval_imgs
      else:
        dir = test_imgs

      os.rename(path, join(dir, img))

  print('Done splitting')

train_stats = {
  'means': {
    'r': 0.,
    'g': 0.,
    'b': 0.
  },
  'std': {
    'r': 0.,
    'g': 0.,
    'b': 0.
  }
}

stat_sample_size = 1000

train_pixels_count = 0

count = 0
for img in os.listdir(train_imgs):
  path = join(train_imgs, img)

  i = Image.open(path)
  arr = np.array(i)

  train_pixels_count += arr.shape[1] * arr.shape[2]

  count += 1
  if count > stat_sample_size:
    break

print(f'Computing stats for {train_pixels_count} total pixels...')

count = 0
for img in os.listdir(train_imgs):
  path = join(train_imgs, img)

  i = Image.open(path)
  arr = np.array(i)

  train_stats['means']['r'] += np.sum(arr[0]/255)/train_pixels_count
  train_stats['means']['g'] += np.sum(arr[1]/255)/train_pixels_count
  train_stats['means']['b'] += np.sum(arr[2]/255)/train_pixels_count

  count += 1
  if count > stat_sample_size:
    break

print(f'Train image means are: r={train_stats["means"]["r"]:.5}, g={train_stats["means"]["g"]:.5}, b={train_stats["means"]["b"]:.5}')

count = 0
for img in os.listdir(train_imgs):
  path = join(train_imgs, img)

  i = Image.open(path)
  arr = np.array(i)

  train_stats['std']['r'] += np.sum((arr[0]/255 - train_stats['means']['r'])**2)/train_pixels_count
  train_stats['std']['g'] += np.sum((arr[1]/255 - train_stats['means']['g'])**2)/train_pixels_count
  train_stats['std']['b'] += np.sum((arr[2]/255 - train_stats['means']['b'])**2)/train_pixels_count

  count += 1
  if count > stat_sample_size:
    break

train_stats['std']['r'] = math.sqrt(train_stats['std']['r'])
train_stats['std']['g'] = math.sqrt(train_stats['std']['g'])
train_stats['std']['b'] = math.sqrt(train_stats['std']['b'])

print(f'Train image standard deviations are: r={train_stats["std"]["r"]:.5}, g={train_stats["std"]["g"]:.5}, b={train_stats["std"]["b"]:.5}')

pickle.dump(train_stats, open('data/train_stats.pickle', 'wb'))

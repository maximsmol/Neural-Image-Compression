from statistics import mean

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_side_by_side(l):
  plt.close()

  my_dpi = 96

  fig = plt.figure(figsize=(2*128*3/my_dpi, 2*128/my_dpi), dpi=my_dpi)
  outer = gridspec.GridSpec(1, 3, wspace=.0, hspace=.0)

  axA = fig.add_subplot(outer[0])
  axB = fig.add_subplot(outer[1])
  codeinner = outer[2].subgridspec(8, 8, wspace=.0, hspace=.0)

  a, b, c = l
  a = a.permute(1, 2, 0)
  b = b.permute(1, 2, 0)
  c = c.permute(1, 2, 0) # 96xW/8xH/8

  axA.imshow(a)
  axB.imshow(b)

  axA.axis('off')
  axB.axis('off')

  for j in range(64):
    if j*3 + 3 >= 96:
      break

    ax = fig.add_subplot(codeinner[j])
    ax.imshow(c[:, :, j*3 : j*3 + 3])
    ax.axis('off')

  return fig

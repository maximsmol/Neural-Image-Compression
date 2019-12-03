import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def show_side_by_side(l):
  plt.close()

  my_dpi = 96

  fig = plt.figure(figsize=(128*3/my_dpi, 128*len(l)/my_dpi), dpi=my_dpi)
  outer = gridspec.GridSpec(len(l), 3, wspace=.0, hspace=.0)

  for i in range(len(l)):
      axA = fig.add_subplot(outer[i*3])
      axB = fig.add_subplot(outer[i*3 + 1])
      codeinner = outer[i*3 + 2].subgridspec(8, 8, wspace=.0, hspace=.0)

      a, b, c = l[i]
      a = a.permute(1, 2, 0)
      b = b.permute(1, 2, 0)
      c = c.permute(1, 2, 0) # 96xW/8xH/8

      axA.imshow(a)
      axB.imshow(b)

      # axA.text(0.5, 0.5, 'HELLO')
      # axB.text(0.5, 0.5, 'HELLO')

      axA.axis('off')
      axB.axis('off')

      for j in range(64):
        if j*3 + 3 >= 96:
          break

        ax = fig.add_subplot(codeinner[j])
        ax.imshow(c[:, :, j*3 : j*3 + 3])
        # ax.text(0.5, 0.5, f'N: {j}')
        ax.axis('off')

  fig.savefig('side-by-side.pdf')

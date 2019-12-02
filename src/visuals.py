import matplotlib
import matplotlib.pyplot as plt

def show_side_by_side(l):
  plt.close()

  fig, axes = plt.subplots(len(l), 2, squeeze=False)
  for i in range(len(l)):
    a, b = l[i]
    a = a.permute(1, 2, 0)
    b = b.permute(1, 2, 0)

    axes[i][0].imshow(a)
    axes[i][1].imshow(b)

  plt.savefig('side-by-side.pdf')

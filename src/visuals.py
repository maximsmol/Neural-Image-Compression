import matplotlib
import matplotlib.pyplot as plt

def show_side_by_side(a, b):
  plt.close()

  a = a.permute(1, 2, 0)
  b = b.permute(1, 2, 0)

  fig, axes = plt.subplots(1, 2)
  axes[0].imshow(a)
  axes[1].imshow(b)

  plt.show()

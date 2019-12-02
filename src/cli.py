import argparse

parser = argparse.ArgumentParser(description='Neural Image Compression')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='set the number of training epochs (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='do not use CUDA even if available')
parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                    help='set the number of batches to wait between checkpoints')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume training from the last checkpoint')
args = parser.parse_args()

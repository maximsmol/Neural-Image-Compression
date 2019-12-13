import argparse

parser = argparse.ArgumentParser(description='Neural Image Compression')
parser.add_argument('--debug-single-batch', action='store_true', default=False,
                    help='stop after a single minibatch for devbug purposes')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='set the target number of training epochs (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='do not use CUDA even if available')
parser.add_argument('--log-interval', type=int, default=60, metavar='N',
                    help='number of seconds to wait between logging training progress')
parser.add_argument('--backup-interval', type=int, default=60*30, metavar='N',
                    help='number of seconds to wait between saving another backup when checkpointing')
parser.add_argument('--save-interval', type=int, default=60*5, metavar='N',
                    help='number of seconds to wait between checkpoints')
parser.add_argument('--restart', action='store_true', default=False,
                    help='always restart training even if there is a checkpoint')
parser.add_argument('--mode', default='train',
                    help='train/eval/infer')
parser.add_argument('image', nargs='?')
args = parser.parse_args()

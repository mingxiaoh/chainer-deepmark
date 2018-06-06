import argparse

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import optimizers as O
from chainer import cuda
import numpy
import six
import json

from deepmark_chainer import net
from deepmark_chainer.utils import timer
from deepmark_chainer.utils import cache


parser = argparse.ArgumentParser(description='Deepmark benchmark for text data.')
parser.add_argument('--predictor', '-p', type=str, default='big-lstm',
                    choices=('small-lstm', 'big-lstm'),
                    help='Network architecture')
parser.add_argument('--seed', '-s', type=int, default=0,
                    help='Random seed')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='The number of iteration to be averaged over.')
parser.add_argument('--seq-length', '-t', type=int, default=50,
                    help='Sequence length')
parser.add_argument("--result", "-r", default=None, help="Result json file path.")
parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU to use. Negative value to use CPU')
parser.add_argument('--cudnn', '-c', action='store_true', help='If this flag is set, cuDNN is enabled.')
parser.add_argument('--cache-level', '-C', type=str, default='none',
                    choices=('none', 'memory', 'disk'),
                    help='This option determines the type of the kernel cache used.'
                    'By default, memory cache and disk cache are removed '
                    'at the beginning of every iteration. '
                    'Otherwise, elapsed times of each iteration are '
                    'measured with corresponding cache enabled. '
                    'If either cache is enabled, this script operates one additional '
                    'iteration for burn-in before measurement. '
                    'This iteration is not included in the mean elapsed time.'
                    'If we do not use GPU, we do not clear cache at all regardless of the value of '
                    'this option.')
parser.add_argument('--batchsize', '-b', type=int, default=64, help='Batchsize')
args = parser.parse_args()
print ("cudnn", args.cudnn)

numpy.random.seed(args.seed)
if args.gpu >= 0:
    cuda.cupy.random.seed(args.seed)

vocab_size = 10

if args.predictor == 'small-lstm':
    predictor = net.small_lstm.SmallLSTM(vocab_size)
elif args.predictor == 'big-lstm':
    predictor = net.big_lstm.BigLSTM(vocab_size)
else:
    raise ValueError('Invalid architector:{}'.format(args.predictor))
model = L.Classifier(predictor)
model.compute_accuracy = False
chainer.config.train = True

if args.gpu >= 0:
    chainer.config.use_cudnn = 'always'
    cuda.get_device(args.gpu).use()
    model.to_gpu()
else:
    chainer.config.use_cudnn = 'never'
    model.to_intel64()
optimizer = O.SGD()
optimizer.setup(model)

xp = cuda.cupy if args.gpu >= 0 else numpy

start_iteration = 0 if args.cache_level is None else -1
forward_time = 0.0
backward_time = 0.0
update_time = 0.0

print('iteration\tforward\tbackward\tupdate (in mseconds)')
for iteration in six.moves.range(start_iteration, args.iteration):
    if args.gpu >= 0:
        cache.clear_cache(args.cache_level)

    # data generation
    data = numpy.random.randint(0, vocab_size,
                                (args.batchsize, args.seq_length)
                                ).astype(numpy.int32)
    data = chainer.Variable(xp.asarray(data))
    label = numpy.random.randint(0, vocab_size,
                                 (args.batchsize, args.seq_length)
                                 ).astype(numpy.int32)
    label = chainer.Variable(xp.asarray(label))

    # forward
    with timer.get_timer(xp) as t:
        loss = model(data, label)
    forward_time_one = t.total_time() * 1000

    # backward
    with timer.get_timer(xp) as t:
        loss.backward()
    backward_time_one = t.total_time() * 1000

    # parameter update
    with timer.get_timer(xp) as t:
        optimizer.update()
    update_time_one = t.total_time() * 1000

    if iteration < 0:
        print('Burn-in\t{}\t{}\t{}'.format(forward_time_one, backward_time_one, update_time_one))
    else:
        print('{}\t{}\t{}\t{}'.format(iteration, forward_time_one, backward_time_one, update_time_one))
        forward_time += forward_time_one
        backward_time += backward_time_one
        update_time += update_time_one

forward_time /= args.iteration
backward_time /= args.iteration
update_time /= args.iteration


forward_sps = args.batchsize * 1000 /forward_time
backward_sps = args.batchsize * 1000 /backward_time
update_sps = args.batchsize * 1000 /update_time
total_sps = args.batchsize * 1000 / (forward_time + backward_time + update_time) 

print('Mean\t{}\t{}\t{}'.format(forward_time, backward_time, update_time))

print('iteration\tforward\tbackward\tupdate\ttotal (sps)')
print('Mean\t{}\t{}\t{}\t{}'.format(forward_sps, backward_sps, update_sps, total_sps))

result = {
        "arch_name": args.predictor,
        "batch_size": args.batchsize,
        "Forward": round(forward_sps,3),
        "Backward": round(backward_sps,3),
        "Uptade": round(update_sps,3),
        "Total": round(total_sps,3)
}
print(result)

if args.result:
   with open(args.result, 'w') as f:
      json.dump(result, f)


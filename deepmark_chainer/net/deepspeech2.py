import six

from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer import link
from chainer.links.connection import convolution_2d as C
from chainer.links.connection import linear as L
from chainer.links.connection import lstm
from chainer.links.connection import gru
from chainer.links.normalization import batch_normalization as B


class StatefulLinearRNN(link.Chain):

    def __init__(self, in_size, out_size, batch_norm_type='upward'):
        super(StatefulLinearRNN, self).__init__(upward=L.Linear(in_size, out_size),
                                                lateral=L.Linear(out_size, out_size))
        if batch_norm_type not in ('none', 'upward', 'lateral', 'output'):
            raise ValueError('Invalid batch_norm_type:{}'.format(batch_norm_type))
        self.batch_norm_type = batch_norm_type

        if batch_norm_type != 'none':
            batch_norm = B.BatchNormalization(out_size)
            self.add_link('batch_norm', batch_norm)

        self.reset_state()

    def to_cpu(self):
        super(StatefulLinearRNN, self).to_cpu()
        if self.h is not None:
            self.h.to_cpu()

    def to_gpu(self, device=None):
        super(StatefulLinearRNN, self).to_gpu(device)
        if self.h is not None:
            self.h.to_gpu(device)

    def reset_state(self):
        self.h = None

    def __call__(self, x):
        h = self.upward(x)
        if self.batch_norm_type == 'upward':
            h = self.batch_norm(h)

        if self.h is not None:
            l = self.lateral(self.h)
            if self.batch_norm_type == 'lateral':
                l = self.batch_norm(l)
            h += l

        if self.batch_norm_type == 'output':
            h = self.batch_norm(h)
        self.h = h
        return self.h


class BRNN(link.Chain):

    def __init__(self, input_dim, output_dim, rnn_unit):
        if rnn_unit == 'LSTM':
            forward = lstm.LSTM(input_dim, output_dim)
            reverse = lstm.LSTM(input_dim, output_dim)
        elif rnn_unit == 'GRU':
            forward = gru.StatefulGRU(output_dim, input_dim)
            reverse = gru.StatefulGRU(output_dim, input_dim)
        elif rnn_unit == 'Linear':
            forward = StatefulLinearRNN(input_dim, output_dim)
            reverse = StatefulLinearRNN(input_dim, output_dim)
        else:
            raise ValueError('Invalid rnn_unit:{}'.format(rnn_unit))
        super(BRNN, self).__init__(forward=forward, reverse=reverse)

    def reset_state(self):
        self.forward.reset_state()
        self.reverse.reset_state()

    def __call__(self, xs):
        N = len(xs)
        x_forward = [self.forward(x) for x in xs]
        x_reverse = [self.reverse(xs[n]) for n
                     in six.moves.range(N - 1, -1, -1)]
        x_reverse.reverse()
        return [x_f + x_r for x_f, x_r in zip(x_forward, x_reverse)]


class ConvBN(link.Chain):

    def __init__(self, *args, **kwargs):
        conv = C.Convolution2D(*args, **kwargs)
        out_channel = len(conv.W.data)
        batch_norm = B.BatchNormalization(out_channel)
        super(ConvBN, self).__init__(conv=conv, batch_norm=batch_norm)

    def __call__(self, x):
        x = self.conv(x)
        return self.batch_norm(x)


class LinearBN(link.Chain):

    def __init__(self, *args, **kwargs):
        linear = L.Linear(*args, **kwargs)
        out_channel = len(linear.W.data)
        batch_norm = B.BatchNormalization(out_channel)
        super(LinearBN, self).__init__(linear=linear, batch_norm=batch_norm)

    def __call__(self, x):
        x = self.linear(x)
        return self.batch_norm(x)


class Sequential(link.ChainList):

    def __call__(self, x, *args, **kwargs):
        for l in self:
            x = l(x, *args, **kwargs)
        return x


class DeepSpeech2(link.Chain):

    def __init__(self, channel_dim=32, hidden_dim=1760, out_dim=29, rnn_unit='Linear'):
        c1 = ConvBN(1, channel_dim, (5, 20), 2)
        c2 = ConvBN(channel_dim, channel_dim, (5, 10), (1, 2))
        convolution = Sequential(c1, c2)

        brnn1 = BRNN(31 * channel_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn2 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn3 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn4 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn5 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn6 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        brnn7 = BRNN(hidden_dim, hidden_dim, rnn_unit=rnn_unit)
        recurrent = Sequential(brnn1, brnn2, brnn3, brnn4,
                               brnn5, brnn6, brnn7)

        fc1 = LinearBN(hidden_dim, hidden_dim)
        fc2 = L.Linear(hidden_dim, out_dim)
        linear = link.ChainList(fc1, fc2)
        super(DeepSpeech2, self).__init__(convolution=convolution,
                                          recurrent=recurrent,
                                          linear=linear)

    def _linear(self, xs):
        ret = []
        for x in xs:
            x = self.linear[0](x)
            x = self.linear[1](x)
            ret.append(x)
        return ret

    def __call__(self, x):
        x = reshape.reshape(x, (len(x.data), 1) + x.data.shape[1:])
        x = self.convolution(x)
        xs = split_axis.split_axis(x, x.data.shape[2], 2)
        for x in xs:
            x.data = self.xp.ascontiguousarray(x.data)
        for r in self.recurrent:
            r.reset_state()
        xs = self.recurrent(xs)
        xs = self._linear(xs)
        return xs

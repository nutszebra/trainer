from __future__ import division, print_function, absolute_import, unicode_literals
import cv2
import six
import chainer
import numpy as np
from chainer import cuda
from chainer import serializers
import chainer.computational_graph as c
from nutszebra_initialization import Initialization as initializer


class Model(chainer.Chain):

    """Some useful functions for chainer model are defined

    Attributes:
        nz_save_model_epoch (int): how many times model is saved
        nz_save_optimizer_epoch (int): how many times optimizer is saved
        nz_lr (float): learning rate
        nz_xp (function): cuda.cupy if gpu, numpy otherwise
        nz_flag_computational_graph (bool): flag for computational_graph
        nz_period (int): period is defined
        nz_count (int): counter
    """

    def __init__(self, **kwargs):
        # initialization for chainer.Chain
        # If you don't initialize, model.to_gpu doesn't work, because no link
        super(Model, self).__init__(**kwargs)
        self.nz_save_model_epoch = 0
        self.nz_save_optimizer_epoch = 0
        self.nz_xp = self._check_cupy()
        self.nz_flag_computational_graph = False

    def remove_link(self, name):
        del self.__dict__[name]
        self._children.remove(name)

    def deregister(self, name):
        self._children.remove(name)

    def register(self, name):
        self._children.append(name)

    def check_gpu(self, gpu):
        """Check cuda.cupy

        Example:

        ::

            gpu = 0
            self.check_gpu(gpu)

        Args:
            gpu (int): gpu id
        """

        if gpu >= 0:
            cuda.get_device(gpu).use()
            self.to_gpu()
            return True
        return False

    @staticmethod
    def _check_cupy():
        """Set xp

        Note:
            cuda.cupy if gpu, numpy otherwise

        Example:

        ::

            self.xp = self._check_cupy()

        Returns:
            cuda.cupy if gpu, numpy otherwise
        """

        try:
            cuda.check_cuda_available()
            return cuda.cupy
        # if gpu is not available, RuntimeError arises
        except RuntimeError:
            return np

    def prepare_input(self, X, dtype=np.float32, volatile=False, xp=None):
        """Prepare input for chainer

        Example:

        ::

            x = np.zeros((100, 100), dtype = np.float32)
            train_x = self.prepare_input(x, dtype=np.float32, volatile=False)
            train_x = self.prepare_input(x, dtype=np.int32, volatile=False)
            test_x = self.prepare_input(x, dtype=np.float32, volatile=True)
            test_x = self.prepare_input(x, dtype=np.int32, volatile=True)

        Args:
            X (list): it can be numpy
            dtype (Optional[np.float32, np.int32...]): np.float32, np.int32....
            volatile (bool): volatile option for chainer variable

        Returns:
            chainer.Variable : you can use this value for input of chainer model
        """

        if xp is None:
            if self.model_is_cpu_mode():
                inp = np.asarray(X, dtype=dtype)
            else:
                inp = self.nz_xp.asarray(X, dtype=dtype)
        else:
            inp = xp.asarray(X, dtype=dtype)
        return chainer.Variable(inp, volatile=volatile)

    def save_computational_graph(self, loss, path='./', name='model.dot'):
        """Save computational graph

        Note:
            computational graph will be generated only once

        Example:

        ::

            self.save_computational_graph(loss, path='./', name='model.dot')

        Returns:
            bool: True if computational graph is generated, False otherwise
        """

        if self.nz_flag_computational_graph is True:
            return False
        else:
            g = c.build_computational_graph((loss, ), remove_split=True)
            with open(path + name, 'w') as o:
                o.write(g.dump())
            self.nz_flag_computational_graph = True

    def model_is_cpu_mode(self):
        """Check whether the mode is cpu mode or not

        Example:

        ::

            answer = model.to_cpu()
            >>> print(answer)
                True

            answer = model.to_gpu()
            >>> print(answer)
                False

        Args:

        Returns:
            bool: True if cpu mode, False otherwise.
        """

        if self.xp == np:
            return True
        else:
            return False

    def load_model(self, path=''):
        serializers.load_npz(path, self)

    def save_model(self, path=''):
        """Save chainer model

        Example:

        ::

            path = './test.model'
            self.save_model(path)

        Args:
            path (str): path

        Returns:
            bool: True if saving successful
        """

        # if gpu_flag is True, switch the model to gpu mode at last
        gpu_flag = False
        # if gpu mode, switch the model to cpu mode temporarily
        if self.model_is_cpu_mode() is False:
            self.to_cpu()
            gpu_flag = True
        # if path is ''
        if path == '':
            path = str(self.save_model_epoch) + '.model'
        self.nz_save_model_epoch += 1
        # increment self.nz_save_model_epoch
        serializers.save_npz(path, self)
        # if gpu_flag is True, switch the model to gpu mode at last
        if gpu_flag:
            self.to_gpu()
        return True

    def save_optimizer(self, optimizer, path=''):
        """Save optimizer model

        Example:

        ::

            path = './test.optimizer'
            self.save_optimizer(optimizer, path)

        Args:
            optimizer (chainer.optimizers): optimizer
            path (str): path

        Returns:
            bool: True if saving successful
        """

        # if path is ''
        if path == '':
            path = str(self.save_optimizer_epoch) + '.optimizer'
        # increment self.nz_save_optimizer_epoch
        self.nz_save_optimizer_epoch += 1
        serializers.save_npz(path, optimizer)
        return True

    @staticmethod
    def heatmap(x, y, confidences, activations, threshold=0.2):
        heatmaps = []
        total_heatmap = []
        for X, Y, confidence, activation in six.moves.zip(x, y, confidences, activations):
            _, _, tmp_heat, tmp_total = _heatmap(X, Y, confidence, activation, threshold=threshold)
            heatmaps.append(tmp_heat)
            total_heatmap.append(tmp_total)
        return (x, y, heatmaps, total_heatmap)

    @staticmethod
    def get_max_activation(total_heatmaps):
        answer = []
        for total_heatmap in total_heatmaps:
            answer.append(np.max(total_heatmap))
        return answer

    @staticmethod
    def get_index_of_max_activation(total_heatmaps):
        max_values = Model.get_max_activation(total_heatmaps)
        indices = []
        for max_value, total_heatmap in six.moves.zip(max_values, total_heatmaps):
            x, y = np.where(total_heatmap == max_value)
            indices.append((x[0], y[0]))
        return indices

    @staticmethod
    def find_minimum_rectangle(coordinate):
        all_y, all_x = list(six.moves.zip(*coordinate))
        x_start, x_end, y_start, y_end = np.min(all_x), np.max(all_x), np.min(all_y), np.max(all_y)
        return ((y_start, x_start), (y_end, x_end))

    @staticmethod
    def find_minimum_rectangles(coordinates):
        answer = []
        for coordinate in coordinates:
            answer.append(Model.find_minimum_rectangle(coordinate))
        return answer

    @staticmethod
    def select_way(way, channel_in, channel_out):
        if way == 'ave':
            n_i = channel_in
            n_i_next = channel_out
        if way == 'forward':
            n_i = channel_in
            n_i_next = None
        if way == 'backward':
            n_i = None
            n_i_next = channel_out
        return n_i, n_i_next

    @staticmethod
    def get_fc_shape(fc):
        channel_out, channel_in = fc.W.data.shape
        return channel_out, channel_in, 1, 1

    @staticmethod
    def get_conv_shape(conv):
        channel_out, channel_in, y_k, x_k = conv.W.data.shape
        return channel_out, channel_in, y_k, x_k

    @staticmethod
    def weight_relu_initialization(link, mean=0.0, relu_a=0.0, way='forward'):
        dim = link.W.data.ndim
        if dim == 2:
            # fc layer
            channel_out, channel_in, y_k, x_k = Model.get_fc_shape(link)
        elif dim == 4:
            # conv layer
            channel_out, channel_in, y_k, x_k = Model.get_conv_shape(link)
        n_i, n_i_next = Model.select_way(way, channel_in * y_k * x_k, channel_out * y_k * x_k)
        # calculate variance
        variance = initializer.variance_relu(n_i, n_i_next, a=relu_a)
        # orthogonal matrix
        w = initializer.orthonorm(mean, variance, (channel_out, channel_in * y_k * x_k), initializer.gauss, np.float32)
        return np.reshape(w, link.W.data.shape)

    @staticmethod
    def bias_initialization(conv, constant=0):
        return initializer.const(conv.b.data.shape, constant=0, dtype=np.float32)


def _heatmap(x, y, confidences, activations, threshold=0.2):
    channel, height, width = x.shape
    heatmaps = []
    max_activation = 0
    for activation, confidence in six.moves.zip(activations, confidences):
        heatmap = np.zeros((height, width))
        activation = confidence * cv2.resize(activation, (width, height), interpolation=cv2.INTER_CUBIC)
        heatmap = heatmap + activation
        heatmaps.append(heatmap)
        max_activation = np.max([max_activation, np.max(heatmap)])
    for heatmap in heatmaps:
        heatmap[np.where(heatmap <= max_activation * threshold)] = 0.0
    total_heatmap = np.zeros((height, width))
    for heatmap in heatmaps:
        total_heatmap = total_heatmap + heatmap
    return (x, y, heatmaps, total_heatmap)

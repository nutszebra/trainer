# To make forward fast
import os
os.environ["CHAINER_TYPE_CHECK"] = "0"
import six
import itertools
import numpy as np
import multiprocessing
import chainer
from chainer import serializers
from chainer import cuda
import nutszebra_log2
import nutszebra_utility
import nutszebra_sampling
import nutszebra_download_cifar10
import nutszebra_data_augmentation_picture
import nutszebra_data_augmentation
import nutszebra_basic_print

try:
    from cupy.cuda import nccl
    _available = True
    chainer.cuda.set_max_workspace_size(chainer.cuda.get_max_workspace_size() * 4)
except ImportError:
    _available = False

Da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
sampling = nutszebra_sampling.Sampling
utility = nutszebra_utility.Utility()

"""
I refered the implementation of MultiprocessParallelUpdate for multiprocessing and nccl.
https://github.com/chainer/chainer/blob/master/chainer/training/updaters/multiprocess_parallel_updater.py
"""


class _Worker(multiprocessing.Process):

    def __init__(self, process_id, pipe, model, gpus, da, batch, master, sampling=sampling()):
        super(_Worker, self).__init__()
        self.process_id = process_id
        self.pipe = pipe
        self.model = model
        self.da = da
        self.device = gpus[process_id]
        self.number_of_devices = len(gpus)
        self.batch = batch
        self.master = master
        self.train_x = master.train_x
        self.train_y = master.train_y
        self.train_batch_divide = master.train_batch_divide
        self.picture_number_at_each_categories = master.picture_number_at_each_categories
        self.parallel = master.parallel_train
        self.sampling = sampling

    def get(self, name):
        return self.__dict__[name]

    def setup(self):
        _, communication_id = self.pipe.recv()
        self.communication = nccl.NcclCommunicator(self.number_of_devices,
                                                   communication_id,
                                                   self.process_id)
        self.model.to_gpu(self.device)

    def run(self):
        dev = cuda.Device(self.device)
        dev.use()
        # build communication via nccl
        self.setup()
        gp = None
        da_args = [self.da() for _ in six.moves.range(self.batch)]
        p = multiprocessing.Pool(self.parallel)
        batch_of_batch = int(float(self.batch) / self.train_batch_divide)
        while True:
            job, data = self.pipe.recv()
            if job == 'finalize':
                dev.synchronize()
                break
            if job == 'update':
                # for reducing memory
                self.model.zerograds()
                indices = list(self.sampling.yield_random_batch_samples(1, self.batch, len(self.train_x), sort=False))[0]
                for ii in six.moves.range(0, len(indices), batch_of_batch):
                    x = self.train_x[indices[ii:ii + batch_of_batch]]
                    t = self.train_y[indices[ii:ii + batch_of_batch]]
                    args = list(six.moves.zip(x, t, da_args))
                    processed = p.starmap(process_train, args)
                    tmp_x, tmp_t = list(zip(*processed))
                    train = True
                    x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.device)
                    t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.device)
                    y = self.model(x, train=train)
                    loss = self.model.calc_loss(y, t) / self.number_of_devices / self.train_batch_divide
                    loss.backward()
                    del x
                    del t
                    del y
                    del loss

                # send gradients of self.model
                gg = gather_grads(self.model)
                null_stream = cuda.Stream.null
                self.communication.reduce(gg.data.ptr,
                                          gg.data.ptr,
                                          gg.size,
                                          nccl.NCCL_FLOAT,
                                          nccl.NCCL_SUM,
                                          0,
                                          null_stream.ptr)
                del gg
                self.model.zerograds()
                # send parameters of self.model
                gp = gather_params(self.model)
                self.communication.bcast(gp.data.ptr,
                                         gp.size,
                                         nccl.NCCL_FLOAT,
                                         0,
                                         null_stream.ptr)
                scatter_params(self.model, gp)
                gp = None


def size_num_grads(link):
    """Count total size of all gradient arrays of a given link
    Args:
        link (chainer.link.Link): Target link object.
    """
    size = 0
    num = 0
    for param in link.params():
        if param.size == 0:
            continue
        size += param.size
        num += 1
    return size, num


def _batch_memcpy():
    return cuda.cupy.ElementwiseKernel(
        'raw T ptrs, raw X info',
        'raw float32 dst',
        '''
            int id_min = id_pre;
            int id_max = num_src;
            while (id_max - id_min > 1) {
                int id = (id_max + id_min) / 2;
                if (i < info[id]) id_max = id;
                else              id_min = id;
            }
            int id = id_min;
            float *src = (float *)(ptrs[id]);
            int i_dst = i;
            int i_src = i;
            if (id > 0) i_src -= info[id];
            dst[i_dst] = 0;
            if (src != NULL) {
                dst[i_dst] = src[i_src];
            }
            id_pre = id;
        ''',
        'batch_memcpy',
        loop_prep='''
                int num_src = info[0];
                int id_pre = 0;
            ''')


def gather_grads(link):
    """Put together all gradient arrays and make a single array
    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    size, num = size_num_grads(link)

    ptrs = np.empty(num, dtype=np.uint64)
    info = np.empty(num + 1, dtype=np.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.grad is not None:
            ptrs[i] = param.grad.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def gather_params(link):
    """Put together all gradient arrays and make a single array
    Args:
        link (chainer.link.Link): Target link object.
    Return:
        cupy.ndarray
    """
    size, num = size_num_grads(link)

    ptrs = np.empty(num, dtype=np.uint64)
    info = np.empty(num + 1, dtype=np.int32)
    info[0] = 0
    i = 0
    for param in link.params():
        if param.size == 0:
            continue
        ptrs[i] = 0  # NULL pointer
        if param.data is not None:
            ptrs[i] = param.data.data.ptr
        info[i + 1] = info[i] + param.size
        i += 1
    info[0] = num

    ptrs = cuda.to_gpu(ptrs, stream=cuda.Stream.null)
    info = cuda.to_gpu(info, stream=cuda.Stream.null)

    return _batch_memcpy()(ptrs, info, size=size)


def scatter_grads(link, array):
    """Put back contents of the specified array to the related gradient arrays
    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_grads()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.grad = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset


def scatter_params(link, array):
    """Put back contents of the specified array to the related gradient arrays
    Args:
        link (chainer.link.Link): Target link object.
        array (cupy.ndarray): gathered array created by gather_params()
    """
    offset = 0
    for param in link.params():
        next_offset = offset + param.size
        param.data = array[offset:next_offset].reshape(param.data.shape)
        offset = next_offset


class TrainCifar10WithMultiGpus(object):

    def __init__(self, model=None, optimizer=None, load_model=None, load_optimizer=None, load_log=None, load_data=None, da=nutszebra_data_augmentation.DataAugmentationNormalizeBigger, save_path='./', epoch=200, batch=128, gpus=(0, 1, 2, 3), start_epoch=1, train_batch_divide=1, test_batch_divide=1, parallel_train=2, parallel_test=16):
        self.model = model
        self.optimizer = optimizer
        self.load_model = load_model
        self.load_optimizer = load_optimizer
        self.load_log = load_log
        self.load_data = load_data
        self.da = da
        self._da = da
        self.save_path = save_path
        self.epoch = epoch
        self.batch = batch
        self.gpus = gpus
        self.start_epoch = start_epoch
        self.train_batch_divide = train_batch_divide
        self.test_batch_divide = test_batch_divide
        self.parallel_train = parallel_train
        self.parallel_test = parallel_test
        # Generate dataset
        self.train_x, self.train_y, self.test_x, self.test_y, self.picture_number_at_each_categories, self.categories = self.data_init()
        # Log module
        self.log = self.log_init()
        # initializing
        self.model_init(model, load_model)
        # create directory
        self.save_path = save_path if save_path[-1] == '/' else save_path + '/'
        utility.make_dir('{}model'.format(self.save_path))
        self.sampling = sampling()
        self._initialized = False
        self._pipes = []
        self._workers = []
        self.communication = None
        self.da_args = [self.da() for _ in six.moves.range(self.batch)]
        self.p_train = multiprocessing.Pool(self.parallel_train)
        self.p_test = multiprocessing.Pool(self.parallel_test)

    def data_init(self):
        dl = nutszebra_download_cifar10.Cifar10()
        data = dl.load_cifar10_data()
        train_x = data['train_x']
        train_y = data['train_y']
        test_x = data['test_x']
        test_y = data['test_y']
        meta = data['meta']
        categories = list(set(train_y.tolist()))
        return (train_x, train_y, test_x, test_y, meta, categories)

    def log_init(self):
        load_log = self.load_log
        log = nutszebra_log2.Log2()
        if load_log is not None:
            log.load(load_log)
        else:
            log({'are': self.categories}, 'categories')
            log({'parameter': len(self.train_x)}, 'train_parameter')
            log({'parameter': len(self.test_x)}, 'test_parameter')
            for i in six.moves.range(len(self.categories)):
                log({'parameter': float((np.array(self.test_y) == i).sum())}, 'test_parameter_{}'.format(i))
            log({'model': str(self.model)}, 'model')
        return log

    @staticmethod
    def model_init(model, load_model):
        if load_model is None:
            print('Weight initialization')
            model.weight_initialization()
        else:
            print('loading {}'.format(load_model))
            serializers.load_npz(load_model, model)

    @staticmethod
    def available():
        return _available

    def _send_message(self, message):
        for pipe in self._pipes:
            pipe.send(message)

    def setup_workers(self):
        # work only once
        if self._initialized:
            return
        self._initialized = True

        self.model.zerograds()
        for i in six.moves.range(1, len(self.gpus)):
            pipe, worker_end = multiprocessing.Pipe()
            worker = _Worker(i, worker_end, self.model, self.gpus, self.da, int(self.batch / len(self.gpus) / self.train_batch_divide), self)
            worker.start()
            self._workers.append(worker)
            self._pipes.append(pipe)

        with cuda.Device(self.gpus[0]):
            self.model.to_gpu(self.gpus[0])
            if len(self.gpus) > 1:
                communication_id = nccl.get_unique_id()
                self._send_message(("set comm_id", communication_id))
                self.communication = nccl.NcclCommunicator(len(self.gpus),
                                                           communication_id,
                                                           0)

    def update_core(self, x, t):
        self._send_message(('update', None))
        with cuda.Device(self.gpus[0]):
            self.model.zerograds()
            # tmp_x = []
            # tmp_t = []
            # for i in six.moves.range(len(x)):
            #     img, info = self.da.train(x[i])
            #     if img is not None:
            #         tmp_x.append(img)
            #         tmp_t.append(t[i])
            args = list(six.moves.zip(x, t, self.da_args))
            processed = self.p_train.starmap(process_train, args)
            tmp_x, tmp_t = list(zip(*processed))
            train = True
            data_length = len(tmp_x)
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpus[0])
            t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.gpus[0])
            y = self.model(x, train=train)
            loss = self.model.calc_loss(y, t) / len(self.gpus)
            loss.backward()
            loss.to_cpu()
            loss = float(loss.data) * data_length

            del x
            del t
            del y

            # NCCL: reduce grads
            null_stream = cuda.Stream.null
            if self.communication is not None:
                # send grads
                gg = gather_grads(self.model)
                self.communication.reduce(gg.data.ptr,
                                          gg.data.ptr,
                                          gg.size,
                                          nccl.NCCL_FLOAT,
                                          nccl.NCCL_SUM,
                                          0,
                                          null_stream.ptr)
                # copy grads, gg, to  self.model
                scatter_grads(self.model, gg)
                del gg
            self.optimizer.update()
            if self.communication is not None:
                gp = gather_params(self.model)
                self.communication.bcast(gp.data.ptr,
                                         gp.size,
                                         nccl.NCCL_FLOAT,
                                         0,
                                         null_stream.ptr)
        return loss

    def finalize(self):
        self._send_message(('finalize', None))

        for worker in self._workers:
            worker.join()

    def train_one_epoch(self):
        self.setup_workers()
        # initialization
        batch_of_batch = int(float(self.batch) / len(self.gpus) / self.train_batch_divide)
        sum_loss = 0
        yielder = self.sampling.yield_random_batch_samples(int(len(self.train_x) / self.batch), int(float(self.batch) / len(self.gpus)), len(self.train_x), sort=False)
        progressbar = utility.create_progressbar(int(len(self.train_x) / self.batch), desc='train', stride=1)
        # train start
        for _, indices in six.moves.zip(progressbar, yielder):
            for ii in six.moves.range(0, len(indices), batch_of_batch):
                x = self.train_x[indices[ii:ii + batch_of_batch]]
                t = self.train_y[indices[ii:ii + batch_of_batch]]
                sum_loss += self.update_core(x, t) * len(self.gpus)
        self.log({'loss': float(sum_loss)}, 'train_loss')
        print(self.log.train_loss())

    def test_one_epoch(self):
        self.setup_workers()
        batch_of_batch = int(self.batch / self.test_batch_divide)
        sum_loss = 0
        sum_accuracy = {}
        sum_5_accuracy = {}
        false_accuracy = {}
        for ii in six.moves.range(len(self.categories)):
            sum_accuracy[ii] = 0
            sum_5_accuracy[ii] = 0
        elements = six.moves.range(len(self.categories))
        for ii, iii in itertools.product(elements, elements):
            false_accuracy[(ii, iii)] = 0
        da = [self._da() for _ in six.moves.range(self.batch)]
        progressbar = utility.create_progressbar(len(self.test_x), desc='test', stride=batch_of_batch)
        for i in progressbar:
            x = self.test_x[i:i + batch_of_batch]
            t = self.test_y[i:i + batch_of_batch]
            tmp_x = []
            tmp_t = []
            args = list(zip(x, t, da))
            processed = self.p_test.starmap(process, args)
            tmp_x, tmp_t = list(zip(*processed))
            data_length = len(tmp_x)
            train = False
            x = self.model.prepare_input(tmp_x, dtype=np.float32, volatile=not train, gpu=self.gpus[0])
            t = self.model.prepare_input(tmp_t, dtype=np.int32, volatile=not train, gpu=self.gpus[0])
            y = self.model(x, train=train)
            # accuracy
            tmp_accuracy, tmp_false_accuracy = self.model.accuracy(y, t)
            for key in tmp_accuracy:
                sum_accuracy[key] += tmp_accuracy[key]
            for key in tmp_false_accuracy:
                false_accuracy[key] += tmp_false_accuracy[key]
            # loss
            loss = self.model.calc_loss(y, t)
            loss.to_cpu()
            sum_loss += float(loss.data) * data_length
        # sum_loss
        self.log({'loss': float(sum_loss)}, 'test_loss')
        # sum_accuracy
        num = 0
        for key in sum_accuracy:
            value = sum_accuracy[key]
            self.log({'accuracy': int(value)}, 'test_accuracy_{}'.format(key))
            num += value
        self.log({'accuracy': int(num)}, 'test_accuracy')
        # false_accuracy
        for key in false_accuracy:
            if key[0] == key[1]:
                pass
            else:
                value = false_accuracy[key]
                self.log({'accuracy': int(value)}, 'test_accuracy_{}_{}'.format(key[0], key[1]))
        # show logs
        sen = [self.log.test_loss(), self.log.test_accuracy(max_flag=True)]
        print('\n'.join(sen))

    def run(self):
        log = self.log
        model = self.model
        optimizer = self.optimizer
        epoch = self.epoch
        start_epoch = self.start_epoch
        save_path = self.save_path
        epoch_progressbar = utility.create_progressbar(epoch + 1, desc='epoch', stride=1, start=start_epoch)
        for i in epoch_progressbar:
            self.train_one_epoch()
            # save model
            model.save_model('{}model/{}_{}.model'.format(save_path, model.name, i))
            optimizer(i)
            self.test_one_epoch()
            log.generate_loss_figure('{}loss.jpg'.format(save_path))
            log.generate_accuracy_figure('{}accuracy.jpg'.format(save_path))
            log.save(save_path + 'log.json')


def process(x, t, da):
    x, info = da.test(x)
    return (x, t)


def process_train(x, t, da):
    x, info = da.train(x)
    return (x, t)

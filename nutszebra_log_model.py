import numpy as np
from chainer import cuda
from functools import wraps
from nutszebra_utility import Utility as utility
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg', warn=False)


def _transfer_to_cpu(value):
    return cuda.to_cpu(value)


def transfer_to_cpu(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if type(value) is not np.ndarray or type(value) is not type(np.float32):
            value = float(_transfer_to_cpu(value))
        return value
    return wrapper


class LogModel(object):

    def __init__(self, model, save_path='./'):
        self.model = model
        self.links = LogModel._track_link(model)
        save_path = save_path if save_path[-1] == '/' else save_path + '/'
        self.stat_path = save_path + 'log/model_stat/'
        self.grad_path = save_path + 'log/grad_stat/'
        self.figure_path = save_path + 'log/'
        self.stat_count = 0
        self.grad_count = 0
        utility.make_dir(self.stat_path)
        utility.make_dir(self.grad_path)

    @staticmethod
    def _children(ele):
        return hasattr(ele, '_children')

    @staticmethod
    def _W(ele):
        return hasattr(ele, 'W')

    @staticmethod
    def _b(ele):
        return hasattr(ele, 'b')

    @staticmethod
    def _link(ele):
        return LogModel._W(ele) or LogModel._b(ele)

    @staticmethod
    def _track_link(model):
        links = {}

        def dfs(name, ele):
            if LogModel._children(ele) is True:
                for child in ele._children:
                    dfs(name + child + '_', ele[child])
            if LogModel._link(ele) is True:
                # conv1/ -> conv
                links[name[:-1]] = ele
        dfs('', model)
        return links

    @staticmethod
    def save(info, path):
        utility.save_json(info, path)

    def save_stat(self):
        LogModel.save(self.extract_stat(), self.stat_path + '{}.json'.format(self.stat_count))
        self.stat_count += 1

    def save_grad(self):
        LogModel.save(self.extract_grad(), self.grad_path + '{}.json'.format(self.grad_count))
        self.grad_count += 1

    def generate_stat_figure(self):
        files = utility.reg_extract(utility.find_files_recursively(self.stat_path), utility.reg_json)
        files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[0]))
        record_mean = defaultdict(list)
        record_max = defaultdict(list)
        record_min = defaultdict(list)
        record_var = defaultdict(list)
        for f in files:
            tmp = utility.load_json(f)
            for key1 in tmp:
                for key2 in tmp[key1]:
                    key = '{}_{}'.format(key1, key2)
                    record_mean[key].append(tmp[key1][key2]['mean'])
                    record_max[key].append(tmp[key1][key2]['max'])
                    record_min[key].append(tmp[key1][key2]['min'])
                    record_var[key].append(tmp[key1][key2]['var'])
        for key in record_mean:
            plt.clf()
            plt.plot(record_mean[key], label='mean')
            plt.plot(record_max[key], label='max')
            plt.plot(record_min[key], label='min')
            plt.draw()
            plt.title(key)
            plt.legend(loc='upper left')
            plt.savefig('{}model_stat_mean_{}.jpg'.format(self.figure_path, key))
        for key in record_var:
            plt.clf()
            plt.plot(record_var[key])
            plt.draw()
            plt.title(key)
            plt.savefig('{}model_stat_var_{}.jpg'.format(self.figure_path, key))
        plt.clf()
        for key in record_mean:
            plt.plot(record_mean[key])
        plt.title('all mean')
        plt.savefig('{}model_stat_all_mean.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_var:
            plt.plot(record_var[key])
        plt.title('all var')
        plt.savefig('{}model_stat_all_var.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_max:
            plt.plot(record_max[key])
        plt.title('all max')
        plt.savefig('{}model_stat_all_max.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_min:
            plt.plot(record_min[key])
        plt.title('all min')
        plt.savefig('{}model_stat_all_min.jpg'.format(self.figure_path))

    def generate_grad_figure(self):
        files = utility.reg_extract(utility.find_files_recursively(self.stat_path), utility.reg_json)
        files = sorted(files, key=lambda s: int(s.split('/')[-1].split('.')[0]))
        record_mean = defaultdict(list)
        record_max = defaultdict(list)
        record_min = defaultdict(list)
        record_var = defaultdict(list)
        for f in files:
            tmp = utility.load_json(f)
            for key1 in tmp:
                for key2 in tmp[key1]:
                    key = '{}_{}'.format(key1, key2)
                    record_mean[key].append(tmp[key1][key2]['mean'])
                    record_max[key].append(tmp[key1][key2]['max'])
                    record_min[key].append(tmp[key1][key2]['min'])
                    record_var[key].append(tmp[key1][key2]['var'])
        for key in record_mean:
            plt.clf()
            plt.plot(record_mean[key], label='mean')
            plt.plot(record_max[key], label='max')
            plt.plot(record_min[key], label='min')
            plt.draw()
            plt.title(key)
            plt.legend(loc='upper left')
            plt.savefig('{}model_stat_mean_{}.jpg'.format(self.figure_path, key))
        for key in record_var:
            plt.clf()
            plt.plot(record_var[key])
            plt.draw()
            plt.title(key)
            plt.savefig('{}model_stat_var_{}.jpg'.format(self.figure_path, key))
        plt.clf()
        for key in record_mean:
            plt.plot(record_mean[key])
        plt.title('all mean')
        plt.savefig('{}model_stat_all_mean.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_var:
            plt.plot(record_var[key])
        plt.title('all var')
        plt.savefig('{}model_stat_all_var.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_max:
            plt.plot(record_max[key])
        plt.title('all max')
        plt.savefig('{}model_stat_all_max.jpg'.format(self.figure_path))
        plt.clf()
        for key in record_min:
            plt.plot(record_min[key])
        plt.title('all min')
        plt.savefig('{}model_stat_all_min.jpg'.format(self.figure_path))

    def yield_links(self):
        for name, link in self.links.items():
            yield (name, link)

    def check_cpu_mode(self):
        return self.model.model_is_cpu_mode()

    @staticmethod
    @transfer_to_cpu
    def calculate_max(array, xp=np):
        return xp.max(array)

    @staticmethod
    @transfer_to_cpu
    def calculate_min(array, xp=np):
        return xp.min(array)

    @staticmethod
    @transfer_to_cpu
    def calculate_mean(array, xp=np):
        return xp.mean(array)

    @staticmethod
    @transfer_to_cpu
    def calculate_var(array, xp=np):
        return xp.var(array)

    @staticmethod
    @transfer_to_cpu
    def calculate_grad_activity(array, xp=np):
        return xp.sum(xp.abs(array))

    @staticmethod
    def _calculate_all(data, xp=np):
        info = {}
        if data is None:
            info['max'] = 0
            info['min'] = 0
            info['mean'] = 0
            info['var'] = 0
        else:
            info['max'] = LogModel.calculate_max(data, xp)
            info['min'] = LogModel.calculate_min(data, xp)
            info['mean'] = LogModel.calculate_mean(data, xp)
            info['var'] = LogModel.calculate_var(data, xp)
        return info

    @staticmethod
    def _extract_stat(link):
        info = {}
        if LogModel._b(link) is True:
            xp = type(link.b.data)
            info['b'] = LogModel._calculate_all(link.b.data, xp)
        if LogModel._W(link) is True:
            xp = type(link.W.data)
            info['W'] = LogModel._calculate_all(link.W.data, xp)
        return info

    def extract_stat(self):
        info = {}
        for name, link in self.yield_links():
            info[name] = LogModel._extract_stat(link)
        return info

    @staticmethod
    def _extract_grad(link):
        info = {}
        if LogModel._b(link) is True:
            xp = type(link.b.data)
            info['b'] = LogModel._calculate_all(link.b.grad, xp)
            # info['b']['activity'] = LogModel.calculate_grad_activity(link.b.grad, xp)
        if LogModel._W(link) is True:
            xp = type(link.W.data)
            info['W'] = LogModel._calculate_all(link.W.grad, xp)
            # info['W']['activity'] = LogModel.calculate_grad_activity(link.W.grad, xp)
        return info

    def extract_grad(self):
        info = {}
        for name, link in self.yield_links():
            info[name] = LogModel._extract_grad(link)
        return info

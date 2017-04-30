import six
import numpy as np
from nutszebra_utility import Utility as utility
import matplotlib
matplotlib.use('Agg', warn=False)
import matplotlib.pyplot as plt


class Log2(object):

    def __init__(self, path=None):
        if path is None:
            self.log = []
        else:
            self.load(path)

    def __call__(self, dic, dic_id='train_loss', ):
        dic['_id'] = dic_id
        self.log.append(dic)

    def search(self, query='train_loss'):
        answer = []
        for dic in self.log:
            if query == dic['_id']:
                answer.append(dic)
        return answer

    def args(self, args):
        name = 'args'
        self(args.__dict__, name)

    @staticmethod
    def divide(array, up='loss', down=1.):
        down = float(down)
        answer = []
        for dic in array:
            answer.append(dic[up] / down)
        return answer

    @staticmethod
    def max(array):
        max_value = np.max(array)
        ind = np.where(np.array(array) == max_value)[-1][-1]
        return (ind, max_value)

    @staticmethod
    def min(array):
        min_value = np.min(array)
        ind = np.where(np.array(array) == min_value)[-1][-1]
        return (ind, min_value)

    @staticmethod
    def prefix(count, value, round_num=5):
        return '|{}: {}'.format(count, round(value, round_num))

    @staticmethod
    def assign_prefix(prefix):
        if prefix == 'loss' or prefix == 'accuracy':
            return Log2.prefix

    @staticmethod
    def show(array, recent=5, max_flag=False, min_flag=False, prefix='loss'):
        answer = []
        begin = 0 if len(array) <= recent else len(array) - recent
        count = begin + 1
        prefix = Log2.assign_prefix(prefix)
        for value in array[begin:]:
            answer.append(prefix(count, value))
            count += 1
        if max_flag is True:
            ind, max_val = Log2.max(array)
            answer.append('|max {}: {}'.format(ind + 1, round(max_val, 5)))
        if min_flag is True:
            ind, min_val = Log2.min(array)
            answer.append('|min {}: {}'.format(ind + 1, round(min_val, 5)))
        return ''.join(answer)

    def train_loss(self, recent=5, max_flag=False, min_flag=False, prefix='loss'):
        parameter = self.search('train_parameter')[0]['parameter']
        return 'train loss ' + Log2.show(Log2.divide(self.search('train_loss'), up='loss', down=parameter), recent=recent, max_flag=max_flag, min_flag=min_flag, prefix=prefix)

    def test_loss(self, recent=5, max_flag=False, min_flag=False, prefix='loss'):
        parameter = self.search('test_parameter')[0]['parameter']
        return 'test loss ' + Log2.show(Log2.divide(self.search('test_loss'), up='loss', down=parameter), recent=recent, max_flag=max_flag, min_flag=min_flag, prefix=prefix)

    def test_accuracy(self, recent=5, max_flag=False, min_flag=False, prefix='accuracy'):
        parameter = self.search('test_parameter')[0]['parameter']
        return 'total accuracy ' + Log2.show(Log2.divide(self.search('test_accuracy'), up='accuracy', down=parameter), recent=recent, max_flag=max_flag, min_flag=min_flag, prefix=prefix)

    def test_5_accuracy(self, recent=5, max_flag=False, min_flag=False, prefix='accuracy'):
        parameter = self.search('test_parameter')[0]['parameter']
        return 'total accuracy ' + Log2.show(Log2.divide(self.search('test_5_accuracy'), up='accuracy', down=parameter), recent=recent, max_flag=max_flag, min_flag=min_flag, prefix=prefix)

    def test_each_accuracy(self, recent=5, max_flag=False, min_flag=False, prefix='accuracy'):
        answer = []
        categories = self.search('categories')[0]['are']
        for i in six.moves.range(len(categories)):
            parameter = self.search('test_parameter_{}'.format(i))[0]['parameter']
            answer.append('accuracy_{} '.format(i) + Log2.show(Log2.divide(self.search('test_accuracy_{}'.format(i)), up='accuracy', down=parameter), recent=recent, max_flag=max_flag, min_flag=min_flag, prefix=prefix))
        return '\n'.join(answer)

    def _dummy_data(self):
        for i in range(1, 20):
            self({'loss': i}, 'train_loss')
        for i in range(1, 20):
            self({'loss': i}, 'test_loss')
        for i in range(1, 20):
            self({'accuracy': i}, 'test_accuracy')
        for i in range(0, 5):
            self({'parameter': 30}, 'test_parameter_{}'.format(i))
            for ii in range(1, 20):
                self({'accuracy': ii}, 'test_accuracy_{}'.format(i))
        self({'parameter': 10}, 'train_parameter')
        self({'parameter': 10}, 'test_parameter')
        self({'are': ['a', 'b', 'c', 'd', 'e']}, 'categories')

    def save(self, path):
        """Save log

        Edited date:
            161014

        Examples:

        ::

            self.save('./log.json')

        Args:
            path (str): it has to be json

        Returns:
            True if successful
        """

        utility.save_json(self.log, path)

    def load(self, path):
        """Load log

        Edited date:
            161014

        Examples:

        ::

            self.load('./log.json')

        Args:
            path (str): it has to be json
        """
        self.log = utility.load_json(path)

    def generate_loss_figure(self, path):
        parameter = self.search('train_parameter')[0]['parameter']
        train_loss = Log2.divide(self.search('train_loss'), up='loss', down=parameter)
        parameter = self.search('test_parameter')[0]['parameter']
        test_loss = Log2.divide(self.search('test_loss'), up='loss', down=parameter)
        plt.clf()
        plt.plot(train_loss, label='train')
        plt.plot(test_loss, label='test')
        plt.title('loss')
        plt.legend()
        plt.draw()
        plt.savefig(path)

    def generate_accuracy_figure(self, path):
        parameter = self.search('test_parameter')[0]['parameter']
        accuracy = Log2.divide(self.search('test_accuracy'), up='accuracy', down=parameter)
        plt.clf()
        plt.plot(accuracy, label='accuracy')
        plt.title('accuracy')
        plt.legend(loc='lower right')
        plt.draw()
        plt.savefig(path)

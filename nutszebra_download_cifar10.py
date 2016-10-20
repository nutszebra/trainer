import six
import numpy as np
import nutszebra_utility as nz
import sys
import pickle


def unpickle(file_name):
    fp = open(file_name, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()
    return data


class Cifar10(object):

    def __init__(self):
        self.utility = nz.Utility()
        self.output_name = 'cifar10.pkl'
        self.url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.downloaded_file = 'cifar-10-python.tar.gz'
        self.untared_file = 'cifar-10-batches-py'
        self.batch_train_file = ['data_batch_' + str(num) for num in six.moves.range(1, 6)]
        self.batch_test_file = 'test_batch'
        self.meta_file = 'batches.meta'
        self.converted_name = 'cifar10.pkl'

    def download_cifar_10(self):
        # if already downloaded and processed, then return True
        if self.converted_name in self.utility.find_files(self.utility.nutszebra_path, affix_flag=True):
            print('Already downloaded')
            return True
        # download file
        print('Downloading: ' + self.downloaded_file)
        self.utility.download_file(self.url, self.utility.nutszebra_path, self.downloaded_file)
        print('Done')
        print('Uncompressing')
        # untar
        self.utility.untar_gz(self.utility.nutszebra_path + '/' + self.downloaded_file)
        print('Done')
        # delete tar.gz file
        self.utility.remove_file(self.downloaded_file)
        # load train file
        print('Loading train data')
        train_x = np.zeros((50000, 3, 32, 32), dtype=np.float32)
        train_y = np.zeros((50000), dtype=np.int32)
        for i, batch_file in enumerate(self.batch_train_file):
            data = unpickle(self.untared_file + '/' + batch_file)
            start = i * 10000
            end = start + 10000
            train_x[start:end] = data['data'].reshape(10000, 3, 32, 32)
            train_y[start:end] = np.array(data['labels'], dtype=np.int32)
        print('Done')
        # load test file
        print('Loading test data')
        test_x = np.zeros((10000, 3, 32, 32), dtype=np.float32)
        test_y = np.zeros((10000), dtype=np.int32)
        data = unpickle(self.untared_file + '/' + self.batch_test_file)
        test_x[:] = data['data'].reshape(10000, 3, 32, 32)
        test_y[:] = np.array(data['labels'], dtype=np.int32)
        print('Done')
        # load meta file
        data = unpickle(self.untared_file + '/' + self.meta_file)
        meta = data['label_names']
        # save loaded data
        print('Saving')
        data = {}
        data['train_x'] = train_x
        data['train_y'] = train_y
        data['test_x'] = test_x
        data['test_y'] = test_y
        data['meta'] = meta
        self.utility.save_pickle(data, self.utility.nutszebra_path + '/' + self.converted_name)

    def check_overlap(self):
        data = self.load_cifar10_data()
        length = len(data['test_x'])
        result = [0] * length
        for i in six.moves.range(length):
            result[i] = np.any(np.all(data['test_x'][i] == data['train_x']))
        return (np.any(result), result)

    def load_cifar10_data(self):
        self.download_cifar_10()
        return unpickle(self.utility.nutszebra_path + '/' + self.converted_name)

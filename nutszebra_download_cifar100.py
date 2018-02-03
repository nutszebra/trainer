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


class Cifar100(object):

    def __init__(self):
        self.utility = nz.Utility()
        self.url = 'http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        self.downloaded_file = 'cifar-100-python.tar.gz'
        self.untared_file = 'cifar-100-python'
        self.converted_name = 'cifar100.pkl'

    def download_cifar_100(self):
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
        data = self.utility.load_pickle('{}/{}'.format(self.untared_file, 'train'), encoding='latin-1')
        train_x = np.array(np.reshape(data['data'], (50000, 3, 32, 32)), dtype=np.float32)
        train_y = np.array(data['fine_labels'], dtype=np.int32)
        # load test file
        print('Loading test data')
        data = self.utility.load_pickle('{}/{}'.format(self.untared_file, 'test'), encoding='latin-1')
        test_x = np.array(np.reshape(data['data'], (10000, 3, 32, 32)), dtype=np.float32)
        test_y = np.array(data['fine_labels'], dtype=np.int32)
        print('Done')
        # load meta file
        data = self.utility.load_pickle('{}/{}'.format(self.untared_file, 'meta'), encoding='latin-1')
        meta = data['fine_label_names']
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
        data = self.load_cifar100_data()
        length = len(data['test_x'])
        result = [0] * length
        for i in six.moves.range(length):
            result[i] = np.any(np.all(data['test_x'][i] == data['train_x']))
        return (np.any(result), result)

    def load_cifar100_data(self):
        self.download_cifar_100()
        return unpickle(self.utility.nutszebra_path + '/' + self.converted_name)

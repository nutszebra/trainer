import cv2
import numpy as np
from time import sleep
import xml.etree.ElementTree as ET
from collections import defaultdict
from nutszebra_utility import Utility as utility


class Assignment(object):

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


class LoadDataset(Assignment):

    filename = ['test.txt', 'train_cls.txt', 'train_loc.txt', 'val.txt']

    def __init__(self, ilsvrc_path, flag_debug=False):
        super(LoadDataset, self).__init__()
        ilsvrc_path = ilsvrc_path[:-1] if ilsvrc_path[-1] == '/' else ilsvrc_path
        for f in self.filename:
            key = f.split('.')[0]
            if 'train' in f:
                self[key] = self._train('{}/ImageSets/CLS-LOC/{}'.format(ilsvrc_path, f), '{}/Data/CLS-LOC/train'.format(ilsvrc_path))
            elif 'val' in f:
                self[key] = self._val('{}/ImageSets/CLS-LOC/{}'.format(ilsvrc_path, f), '{}/Data/CLS-LOC/val'.format(ilsvrc_path), '{}/Annotations/CLS-LOC/val'.format(ilsvrc_path))
            elif 'test' in f:
                self[key] = self._test('{}/ImageSets/CLS-LOC/{}'.format(ilsvrc_path, f), '{}/Data/CLS-LOC/test'.format(ilsvrc_path))
        self.debug(flag_debug)

    def debug(self, flag=False):
        # no debug
        if flag is False:
            return
        for name in ['train_cls', 'train_loc', 'val', 'test']:
            print('checking {}'.format(name))
            sleep(2)
            if 'train' in name:
                counter = 0
                for key in self[name]:
                    counter += len(self[name][key])
                    print('{}: {}'.format(key, len(self[name][key])))
                    index = int(np.random.randint(len(self[name][key])))
                    # load one pictru randomly
                    if cv2.imread(self[name][key][index]) is None:
                        raise Exception('{} is invalid'.format(self[name][key][index]))
                print('{} in total: {}'.format(name, counter))
            else:
                print('{}: {}'.format(name, len(self[name])))
                index = int(np.random.randint(len(self[name])))
                # load one pictru randomly
                if cv2.imread(self[name][index]) is None:
                        raise Exception('{} is invalid'.format(self[name][index]))

    @staticmethod
    def _train(path, prefix):
        train = defaultdict(list)
        for line in utility.yield_text(path):
            category, name = line.split(' ')[0].split('/')
            train[category].append('{}/{}/{}.JPEG'.format(prefix, category, name))
        return train

    @staticmethod
    def _val(path, prefix1, prefix2):
        val = defaultdict(list)
        for line in utility.yield_text(path):
            name = line.split(' ')[0]
            for obj in ET.parse('{}/{}.xml'.format(prefix2, name)).getroot().findall('object'):
                key = obj.find('name').text
                break
            val[key].append('{}/{}.JPEG'.format(prefix1, name))
        return val

    @staticmethod
    def _test(path, prefix):
        not_train = []
        for line in utility.yield_text(path):
            name = line.split(' ')[0]
            not_train.append('{}/{}.JPEG'.format(prefix, name))
        return not_train

import numpy as np
from nutszebra_utility import Utility as utility


class LoadDataset(object):

    def __init__(self):
        pass

    @staticmethod
    def load_uni_label(path):
        data = utility.load_json(path)
        categories = sorted(data['test'].keys())
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        picture_number_at_each_categories = []
        for i, category in enumerate(categories):
            # train
            number_of_picture = len(data['train'][category])
            picture_number_at_each_categories.append(number_of_picture)
            train_x = train_x + data['train'][category]
            train_y = train_y + [i] * number_of_picture
            # test
            number_of_picture = len(data['test'][category])
            test_x = test_x + data['test'][category]
            test_y = test_y + [i] * number_of_picture
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        return (train_x, train_y, test_x, test_y, picture_number_at_each_categories, categories)

    @staticmethod
    def load_uni_label_and_unlabel(path):
        train_x, train_y, test_x, test_y, picture_number_at_each_categories, categories = LoadDataset.load_uni_label(path)
        unlabeled_x = np.array(utility.load_json(path)['unlabeled'])
        return (train_x, train_y, unlabeled_x, test_x, test_y, picture_number_at_each_categories, categories)

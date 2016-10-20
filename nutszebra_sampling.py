import six
import types
import itertools
import numpy as np


class Sampling(object):

    """Some useful functions for sampling are defined

    Attributes:
        input_configuration (dict): store sampling yielders
        _backup_input_configuration (dict): copy of self.input_configuration
        _dummy_type_tee (itertools.tee): type of itertools.tee
    """

    def __init__(self):
        self.input_configuration = {}
        self._backup_input_configuration = {}
        self._dummy_type_tee = type(self.dummy_type_tee())

    @staticmethod
    def dummy_type_tee():
        """Give itertools.tee(yielder)[0]

        Edited date:
            160704

        Test:
            160704

        Returns:
            itertools.tee: this is used self.type_generator_or_tee
        """
        def dummy():
            yield None
        copy1, copy2 = itertools.tee(dummy())
        return copy2

    @staticmethod
    def type_generator_or_tee(generator):
        """Check generator is generator or itertools.tee

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            >>> print(self.type_generator_or_tee(p))
            True

            >>> print(self.type_generator_or_tee(1))
            False

        Returns:
            True if yielder is generator or itertools.tee, False otherwise
        """
        if isinstance(generator, types.GeneratorType):
            return True
        if isinstance(generator, type(Sampling.dummy_type_tee())):
            return True
        return False

    def keys(self):
        """Give keys of self.input_configuration

        Edited date:
            160704

        Test:
            160704

        Returns:
            list: keys of self.input_configuration
        """

        return self.input_configuration.keys()

    def set(self, func_yields, keys):
        """Set yielders

        Edited date:
            160704

        Test:
            160704

        Note:
            usage is written in Example. Normaly you just set yielders by self.set, then call self.sample_from for sampling. You can reset yielders by self.reset, but be aware that resetted random samplig yielders is exactyly same as before.

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            p1 = a.yield_continuous_random_batch_samples(3, 3, 10)
            p2 = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set([p1, p2], ['test1', 'test2'])
            for ele in self.sample_from('test1'):
                >>> print(ele)
            [1 6 0]
            [2 7 1]
            [3 8 2]

            for ele in self.sample_from('test2'):
                >>> print(ele)
            [1 5 3]
            [2 6 4]
            [3 7 5]

        Args:
            func_yields Optional([list, generator, tee]): generator to set
            keys Optional([list, str]): the key of generator

        Returns:
            True if yielders are setted, False otherwise
        """
        # if func_yields and keys are stored into list
        if isinstance(func_yields, list) and isinstance(keys, list):
            for func_yield, key in six.moves.zip(func_yields, keys):
                # generator or tee?
                if self.type_generator_or_tee(func_yield):
                    # copy generator by using itertools.tee
                    self.input_configuration[key], self._backup_input_configuration[key] = itertools.tee(func_yield)
            return True
        # if func_yields is just generator or tee
        if self.type_generator_or_tee(func_yields):
            # copy generator by using itertools.tee
            self.input_configuration[keys], self._backup_input_configuration[keys] = itertools.tee(func_yields)
            return True
        return False

    def delete_all(self):
        """Delete all self.input_configuration

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            self.delete_all()

        Returns:
            True if all yielders are deleted, False otherwise
        """
        self.input_configuration = {}
        self._backup_input_configuration = {}
        return True

    def delete(self, key):
        """Delete self.input_configuration[key]

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            self.delete('test')

        Returns:
            True if the yielder is deleted, False otherwise
        """
        if key in self.input_configuration:
            self.input_configuration.pop(key)
            self._backup_input_configuration.pop(key)
            return True
        return False

    def reset_all(self):
        """Reset all self.input_configuration by self._backup_input_configuration

        Edited date:
            160704

        Test:
            160704

        Note:
            usage is written in Example. Normaly you just set yielders by self.set, then call self.sample_from for sampling. You can reset yielders by self.reset_all, but be aware that resetted random samplig yielders is exactyly same as before.

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

            self.reset_all()
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

        Returns:
            True if successful
        """
        for key in self._backup_input_configuration:
            self.set(self._backup_input_configuration[key], key)
        return True

    def reset(self, key):
        """Reset self.input_configuration[key] by self._backup_input_configuration[key]

        Edited date:
            160704

        Test:
            160704

        Note:
            usage is written in Example. Normaly you just set yielders by self.set, then call self.sample_from for sampling. You can reset yielders by self.reset, but be aware that resetted random samplig yielders is exactyly same as before.

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

            self.reset('test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

        Args:
            key: key for self.input_configuration

        Returns:
            True if self.input_configuration[key] is resetted, False otherwise
        """
        if key in self._backup_input_configuration:
            self.set(self._backup_input_configuration[key], key)
            return True
        return False

    def sample_from(self, key):
        """Return yielder

        Edited date:
            160704

        Test:
            160704

        Note:
            usage is written in Example. Normaly you just set yielders by self.set, then call self.sample_from for sampling. You can reset yielders by self.reset_all, but be aware that resetted random samplig yielders is exactyly same as before.

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 3, 10)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

            self.reset_all()
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 0 6]
            [2 1 7]
            [3 2 8]

            for ele in self.sample_from('test'):
                >>> print(ele)

        Args:
            key: key for self.input_configuration

        Returns:
            yield : yielder is returned
        """
        return self.input_configuration[key]

    @staticmethod
    def yield_batch_samples(end, start=0, stride=1):
        """Yield range(start, end, stride)

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = a.yield_batch_samples(3)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
            >>>  print(ele)
              0
              1
              2

            p = a.yield_batch_samples(10, start=1, stride=3)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
            >>> print(ele)
              1
              4
              7

        Args:
            start (int): index starts from this
            end (int): index ends this
            stride (int): stride

        Yields:
            int: range(start, end, stride)
        """
        for i in six.moves.range(start, end, stride):
            yield i

    @staticmethod
    def yield_equal_interval_batch_samples(batch, sample_length):
        """Give batch samples with equal interval

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3,10)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [0 3 6]
            [1 4 7]
            [2 5 8]

        Args:
            batch (int): batch number
            sample_length (int): lengtgh of sample list

        Yields:
            numpy.ndarray: sampled indices
        """
        batch_length = int(sample_length // batch)
        samples = Sampling.batch_sample_equal_interval(batch, sample_length)
        for i in six.moves.range(batch_length):
            yield samples + int(i)

    @staticmethod
    def yield_continuous_random_batch_samples(batch, epoch, sample_length):
        """Give random batch indices continuously

        Edited date:
            160704

        Test:
            160704

        Note:
            sampling starts from sample_length - batch_length

        Example:

        ::

            p = a.yield_continuous_random_batch_samples(3, 5, 100)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [74 67 43]
            [75 68 44]
            [76 69 45]
            [77 70 46]
            [78 71 47]

        Args:
            batch (int): batch number
            epoch (int): how many times self.yield_random_batch_samples yield
            sample_length (int): lengtgh of sample list

        Yields:
            numpy.ndarray: sampled indices
        """
        samples = Sampling.pick_random_permutation(batch, sample_length - epoch)
        for i in six.moves.range(epoch):
            yield samples + int(i)

    @staticmethod
    def yield_random_batch_samples(epoch, batch, sample_length, sort=False):
        """Give batch indices randomly

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            p = self.yield_random_batch_samples(3, 5, 10, sort=False)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [7 0 3 2 6]
            [8 1 9 7 5]
            [0 6 9 5 2]

            self.delete_all()
            p = self.yield_random_batch_samples(3, 5, 10, sort=True)
            self.set(p, 'test')
            for ele in self.sample_from('test'):
                >>> print(ele)
            [1 3 5 8 9]
            [0 1 7 8 9]
            [2 3 4 5 6]

        Args:
            epoch (int): how many times self.yield_random_batch_samples yields
            batch (int): batch number
            sample_length (int): lengtgh of sample list
            sort (bool): sorted list will be returned if True, otherwise False

        Yields:
            numpy.ndarray: sampled indices
        """

        epoch = int(epoch)
        batch = int(batch)
        sample_length = int(sample_length)
        sort = bool(sort)
        for i in six.moves.range(epoch):
            yield Sampling.pick_random_permutation(batch, sample_length, sort=sort)

    @staticmethod
    def yield_random_batch_from_category(epoch, number_of_picture_at_each_categories, pick_number, sequence=True, shuffle=True):
        """Yield batch that samples equally over imbalanced category randomly

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            >>> print(list(sample.yield_random_batch_from_category(3,[3,3,3],5,sequence=False)))
                 # [[0, 1], [0, 2], [2]] means sampled index from [category[0], category[1], category[2]]
                 [[[0, 1], [0, 2], [2]], [[2, 0], [2], [0, 1]], [[1, 0], [2], [2, 0]]]
            >>> print(list(sample.yield_random_batch_from_category(3,[3,3,3],5,sequence=True, shuffle=False)))
                 # 0~2 is category[0]
                 # 3~5 is category[1]
                 # 6~8 is category[2]
                 # shuffle is False, thus the order is kept
                 [[1, 1, 4, 4, 8], [0, 3, 3, 5, 8], [1, 2, 4, 3, 7]]
            >>> print(list(sample.yield_random_batch_from_category(3,[3,3,3],5,sequence=True, shuffle=True)))
                 [[6, 6, 8, 7, 1], [8, 7, 5, 4, 0], [6, 0, 2, 7, 5]]

        Args:
            epoch (int): how many time this function yield
            number_of_picture_at_each_categories (list): it contains the number of sample at each category. It is expected to be imbalance.
            pick_number (int): how many samples you need
            sequence (bool): If True, categories are considered to be sequent
            shuffle (bool): If True, indices are shuffled. However, if sequence is False, shuffle does not occur, because it is useless

        Returns:
            numpy.ndarray: randomyly sampled permutation
        """

        # they have to be int
        epoch = int(epoch)
        pick_number = int(pick_number)
        # total category
        number_of_categories = len(number_of_picture_at_each_categories)
        # yield epoch times
        for i in six.moves.range(epoch):
            # Be carefull, [[]] * 3 causes problem
            # sample[0] means the sample of category[0]
            sample = [[] for _ in six.moves.range(number_of_categories)]
            for ii in six.moves.range(pick_number):
                # select category randomly
                index = Sampling.pick_random_permutation(2, number_of_categories)[0]
                # get one sample from selected category
                sample[index].append(Sampling.pick_random_permutation(1, number_of_picture_at_each_categories[index])[0])
            if sequence is True:
                sequence_sample = []
                base_number = 0
                for index in six.moves.range(number_of_categories):
                    for sample_index in sample[index]:
                        sequence_sample.append(base_number + sample_index)
                    base_number += number_of_picture_at_each_categories[index]
                if shuffle is True:
                    # np.random.shuffle is destructive
                    np.random.shuffle(sequence_sample)
                yield sequence_sample
            else:
                yield sample

    @staticmethod
    def pick_random_permutation(pick_number, sample_number, sort=False):
        """Give random permutation

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            answer = self.pick_random_permutation(3, 10)
            >>> print(answer)
            array([9, 5, 8])

            answer = self.pick_random_permutation(10, 100, sort=True)
            >>> print(answer)
            array([2, 8, 22, 39, 44, 46, 58, 69, 85, 99])

        Args:
            pick_number (int): how many samples you need
            sample_number (int): Total number of sample
            sort (bool): sorted list will be returned if True, otherwise False

        Returns:
            numpy.ndarray: randomyly sampled permutation
        """

        pick_number = int(pick_number)
        sample_number = int(sample_number)
        sort = bool(sort)
        if sort:
            return np.sort(np.random.permutation(sample_number)[:pick_number])
        else:
            return np.random.permutation(sample_number)[:pick_number]

    @staticmethod
    def batch_sample_equal_interval(batch, sample_length):
        """Give indices with equal interval

        Edited date:
            160704

        Test:
            160704

        Example:

        ::

            answer = self.batch_sample_equal_interval(3, 10)
            >>> print(answer)
            array([0, 3, 6])

            answer = self.batch_sample_equal_interval(3, 11)
            >>> print(answer)
            array([0, 3, 6])

            answer = self.batch_sample_equal_interval(3, 12)
            >>> print(answer)
            array([0, 4, 8])

        Args:
            batch (int): total batch number
            sample_length (int): divide the list that len(list) is sample_length into equal interval

        Returns:
            numpy.ndarray: indices with equal interval
        """
        jump = int(sample_length // batch)
        indices = np.array(list(six.moves.range(int(batch))), np.int)
        indices = indices * jump
        return indices

    @staticmethod
    def yield_siamese_dataset(data, howmany, num_positive, num_negative):
        keys = list(data.keys())
        keys.sort()
        for i in six.moves.range(howmany):
            anchor = None
            positive = []
            negative = []
            identity_index = Sampling.pick_random_permutation(1, len(keys))[0]
            positive_index = list(Sampling.pick_random_permutation(num_positive + 1, len(data[keys[identity_index]]), sort=True))
            anchor_index = Sampling.pick_random_permutation(1, num_positive + 1)[0]
            anchor = (identity_index, positive_index.pop(anchor_index))
            positive = [(identity_index, index) for index in positive_index]
            for ii in six.moves.range(num_negative):
                while 1:
                    neg_identity = Sampling.pick_random_permutation(1, len(keys))[0]
                    if neg_identity is not identity_index:
                        break
                negative_index = Sampling.pick_random_permutation(1, len(data[keys[neg_identity]]))[0]
                negative.append((neg_identity, negative_index))
            yield (anchor, positive, negative)

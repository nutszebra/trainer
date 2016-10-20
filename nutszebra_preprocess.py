import numpy as np


class Preprocess(object):

    """Some useful functions for preprocessing are defined

    Attributes:
        normalized_gaussian (dict): some cdf values of normalized gaussian distribution are stored
    """

    def __init__(self):
        """Initialization of Preprocess

        """
        # cdf(-1.64486): 0.049999342715004093
        # cdf(1.64486): 0.95000065728499594
        # cdf(-1.28155157): 0.09999999921808489
        # cdf(1.28155157): 0.90000000078191511
        # cdf(-1.0364333895): 0.14999999999855201
        # cdf(1.0364333895): 0.85000000000144804
        self.normalized_gaussian_cdf = {}
        self.normalized_gaussian_cdf['5%'] = -1.64486
        self.normalized_gaussian_cdf['95%'] = 1.64486
        self.normalized_gaussian_cdf['10%'] = -1.28155157
        self.normalized_gaussian_cdf['90%'] = 1.28155157
        self.normalized_gaussian_cdf['15%'] = -1.0364333895
        self.normalized_gaussian_cdf['85%'] = 1.0364333895

    def count_met_conditions(self, x, condition=True):
        """count the number of elements that satisfies the condition

        Note:
            True and 1 are considered as the same condition, samely False and 0 are

        Example:
            x = [True] * 5 + [False] * 10
            condition = False
            answer = self.count_met_conditions(x, condition=condition)
            >>> print(answer)
            10

        Args:
            x (list): it can be numpy
            condition (Optional[bool]): condition, it can be number also

        Returns:
            int: the number of elements that satisfies the condition
        """
        x = np.array(x)
        return len(np.where(x == condition)[0])

    def convert_to_normalized_gaussian(self, x):
        """convert x to normalized gaussian distribution

        Example:
            x = range(10)
            answer = self.convert_to_normalized_gaussian(x)
            >>> print(answer)
            (array([-1.5666989 ,-1.21854359, -0.87038828, -0.52223297,
                    -0.17407766, 0.17407766, 0.52223297, 0.87038828,
                    1.21854359, 1.5666989 ]),
            4.5,
            2.8722813232690143)

        Args:
            x (list): it can be numpy

        Returns:
            tuple: z, average, standard deviation
        """
        x = np.array(x)
        average = np.mean(x)
        std = np.std(x)
        z = (x - average) / std
        return (z, average, std)

    def divide_into_two(self, x, ratio=0.8):
        """divide list into two

        Note:
            the order of element inside list is conserved

        Examples:
            x = range(10)
            self.divide_into_two(10, ratio=0.5)
            >>> print(files)
            ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])

        Args:
            x (list): it can be numpy
            ratio (float): ratio of list size

        Returns:
            tuple: divided list
        """
        index = int(len(x) * float(ratio))
        return (x[:index], x[index:])

    def divide_into_two_randomly(self, x, ratio=0.8):
        """divide list into two randomly

        Note:
            divided lists are unique

        Examples:
            x = range(10)
            self.divide_into_two(10, ratio=0.5)
            >>> print(files)
            (array([9, 5, 2, 1, 3]), array([8, 7, 4, 0, 6]))

        Args:
            x (list): it can be numpy
            ratio (float): ratio of list size

        Returns:
            tuple: divided list
        """
        indices = np.random.permutation(len(x))
        index = int(len(x) * float(ratio))
        x = np.array(x)
        return (x[indices[:index]], x[indices[index:]])

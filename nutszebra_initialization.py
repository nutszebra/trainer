import numpy as np


class Initialization(object):

    """Some useful functions for initialization

    Attributes:
    """

    def __init__(self):
        pass

    @staticmethod
    def gauss(sizes, variance=0.01, mean=0.0, dtype=np.float32):
        """Give numpy.ndarray back that are initialized with the gaussian distribution

        Edited date:
            160419

        Test:
            160419

        Example:

        ::

            answer = self.gauss((1000, 1000), variance=0.01, mean=0)
            >>> print(np.var(answer))
                0.010005431337713115
            >>> print(np.mean(answer))
                -0.00019472348389183342

        Args:
            sizes (tuple): output size
            variance (Optional[float or int]): the variance of the gaussian distribution
            mean (Optional[float or int]): the mean of the gaussian distribution
            dtype (Optional[np.float64, np.float32, np.int32...]): data type

        Returns:
            numpy.ndarray: values that are sampled from the gaussian distribution
        """

        std = np.sqrt(variance)
        return np.array(np.random.normal(mean, std, sizes), dtype=dtype)

    @staticmethod
    def uniform(sizes, variance=0.01, mean=0.0, dtype=np.float32):
        """Give numpy.ndarray back that are initialized with the uniform distribution

        Edited date:
            160419

        Test:
            160419

        Note:

        | Var[U[a, b]] = (b - a)^2 / 12
        |   Var: the variance
        |   U: the uniform distribution
        |   a, b: the range of the uniform distribution
        | Assume the symmetric uniform distribution,
        |   b = -a
        | Then, b has to be:
        |   b = sqrt(3*Var)

        Example:

        ::

            answer = self.uniform((1000, 1000), variance=0.01, mean=0)
            >>> print(np.var(answer))
                0.010002619805106435
            >>> print(np.mean(answer))
                1.2950964746309503e-05

            answer = self.uniform((1000, 1000), variance=0.01, mean=1)
            >>> print(np.var(answer))
                0.010006011137365682
            >>> print(np.mean(answer))
                1.0001240680416341


        Args:
            sizes (tuple): output size
            variance (Optional[float or int]): the variance of the uniform distribution
            mean (Optional[float or int]): the mean of the uniform distribution
            dtype (Optional[np.float64, np.float32, np.int32...]): data type

        Returns:
            numpy.ndarray: values that are sampled from the uniform distribution
        """

        high = np.sqrt(3 * variance)
        low = - high
        return np.array(mean + np.random.uniform(low, high, sizes), dtype=dtype)

    @staticmethod
    def orthonorm(mean, variance, sizes, random, dtype=np.float32):
        """Give numpy.ndarray back that are initialized with the orthnorm initialization

        Paper: Exact solutions to the nonlinear dynamics of learning in deep linear neural networks

        Paper url: http://arxiv.org/abs/1312.6120

        Edited date:
            160419

        Test:
            160419

        Note:

        | Weights are orthogonal matrix.
        | Consider the case of CNN,
        | Let input channel be n_i, output channel be n_o and convolutional filter size be k*k.
        | Then sizes has to be (n_o, k*k*n_i).
        | Consider the case of fully-connected layer,
        | Let the number of resposes from layer l-1 be z and the number of neurons at layer l be w.
        | Then sizes has to be (w, z).

        Example:

        ::

            mean = 0
            variance = 0.01
            sizes = (1000, 1000)
            answer = self.orthonorm(mean, variance, sizes, self.gauss)
            >>> print(np.var(answer))
                0.01
            >>> print(np.mean(answer))
                2.915889751875511e-18

            # check whether answer is orthogonal matrix or not
            answer_unit = answer.dot(answer.T)
            scale = answer_unit[0][0]
            unit_matrix = scale * np.identity(answer_unit.shape[0])
            unit_or_not = np.all(np.isclose(answer_unit - unit_matrix, 0, atol=1.0e-2))
            >>> print(unit_or_not)
                True

        Args:
            mean (int or float): the mean of weights
            variance (int or float): the variance of weights
            sizes (tuple): output size
            random (self.gauss, self.uniform): the way of randomization
            dtype (Optional[np.float64, np.float32, np.int32...]): data type

        Returns:
            numpy.ndarray: values that are initialized with orthonorm initialization
        """

        u, _, v = np.linalg.svd(random(sizes, variance=variance, mean=mean), full_matrices=False)
        # pick the one with the correct shape
        q = u if u.shape == sizes else v
        q = q / np.sqrt(np.var(q)) * np.sqrt(variance)
        return np.array(q + (mean - np.mean(q)), dtype=dtype)

    @staticmethod
    def const(sizes, constant=0, dtype=np.float32):
        """Give numpy.ndarray back that are initialized with the constant number

        Edited date:
            160419

        Test:
            160419

        Example:

        ::

            answer = self.const((1000, 1000), constant=10)
            >>> print(np.var(answer))
            array([[10., 10., 10., ...., 10., 10., 10.],
                   [10., 10., 10., ...., 10., 10., 10.],
                   [10., 10., 10., ...., 10., 10., 10.],
                   ...,
                   [10., 10., 10., ...., 10., 10., 10.],
                   [10., 10., 10., ...., 10., 10., 10.],
                   [10., 10., 10., ...., 10., 10., 10.]], dtype=float32)

            >>> print(np.mean(answer))
                10.0
            answer = self.uniform((1000, 1000), variance=0.01, mean=1)
            >>> print(np.var(answer))
                0.0

        Args:
            sizes (tuple): output size
            const (Optional[float]): constant number
            dtype (Optional[np.float64, np.float32, np.int32...]): data type

        Returns:
            numpy.ndarray: values that are sampled from uniform distirbution
        """

        return constant * np.ones(sizes, dtype=dtype)

    @staticmethod
    def variance_lecun(node_input):
        """give back the variance for lecun initialization

        Paper: Efficient Backprop

        Paper url: http://goo.gl/FXsRAq

        Edited date:
            160419

        Test:
            160419

        Example:

        ::

            answer = self.lecun(100)
            >>> print(np.var(answer))
                0.01

        Args:
            node_input (int): the number of connections feeding into the node

        Returns:
            float: the variance for lecun initialization
        """

        return 1.0 / node_input

    @staticmethod
    def variance_xavier(node_input, node_input_next):
        """give back the variance for xavier initialization

        Paper: Understanding the difficulty of training deep feedforward neural networks

        Paper url: http://goo.gl/dJyx2w

        Edited date:
            160419

        Test:
            160419

        Note:

        | The detail is written here: https://goo.gl/0y7HE2

        Example:

        ::

            node_input = 1000.0
            node_input_next = 2000.0
            answer = self.xavier(node_input, None)
            >>> print(np.var(answer))
                0.001

            answer = self.xavier(None, node_input_next)
            >>> print(np.var(answer))
                0.0005

            answer = self.xavier(node_input, node_input_next)
            >>> print(np.var(answer))
                0.0006666666666666666

        Args:
            node_input (int): the number of connections feeding into the node at layer l
            node_input_next (int): the number of connections feeding into the node at layer l+1

        Returns:
            float: the variance for xavier initialization
        """

        if node_input is not None:
            if node_input_next is not None:
                # averaged case
                return 2.0 / (node_input + node_input_next)
            else:
                # forward case
                return 1.0 / node_input
        else:
                # backward case
                return 1.0 / node_input_next

    @staticmethod
    def variance_relu(node_input, node_input_next, a=0.0):
        """give back the variance for ReLU initialization

        Paper: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

        Paper url: http://arxiv.org/abs/1502.01852

        Edited date:
            160419

        Test:
            160419

        Note:

        | The detail is written here: https://goo.gl/3uBlBg
        | Typically for PReLU, a is initialized with 0.25

        Example:

        ::

            node_input = 1000.0
            node_input_next = 2000.0
            answer = self.xavier(node_input, None)
            >>> print(np.var(answer))
                0.001

            answer = self.xavier(None, node_input_next)
            >>> print(np.var(answer))
                0.0005

            answer = self.xavier(node_input, node_input_next)
            >>> print(np.var(answer))
                0.0006666666666666666

        Args:
            node_input (int): the number of connections feeding into the node at layer l
            node_input_next (int): the number of connections feeding into the node at layer l+1
            a (float): the parameter for PReLU. If a is 0, then it means ReLU.

        Returns:
            float: the variance for relu initialization
        """

        if node_input is not None:
            if node_input_next is not None:
                # averaged case
                return 4.0 / ((1 + a ** 2) * (node_input + node_input_next))
            else:
                # forward case
                return 2.0 / ((1 + a ** 2) * node_input)
        else:
                # backward case
                return 2.0 / ((1 + a ** 2) * node_input_next)

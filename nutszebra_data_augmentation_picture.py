import six
import cv2
import inspect
import itertools
import numpy as np
from functools import wraps
# from sklearn.decomposition import PCA
from scipy.ndimage.interpolation import rotate as imrotate
from nutszebra_sampling import Sampling as sampling
from nutszebra_preprocess_picture import PreprocessPicture as preprocess
from nutszebra_basic_dictionary import NutszebraDictionary as dict_n


def execute_based_on_probability(func):
    """Decorator to execute a function based on probability

    Edited date:
        160707

    Test:
        |    2. str or numpy.ndarray: The decorator treat it as the input x of func, thus set the first argument as self.x and pass it to func directly with **kwargs. You can give info as the second argumnet and in that case, the second argument becomes self.info. If you don't give info, new info is generated and setted as self.info.
        |    3. Nothing or None: The first argument is treated as 1.0.

    Args:
        __no_record (bool): If True, this decorator does not do anything
        x_or_probability Optional([str, numpy.ndarray, int, float, None]): read Note
        info [nutszebra_basic_dictionary.NutszebraDictionary]: read Note

    Returns:
        nutszebra_data_augmentation_picture.DataAugmentationPicture: return self
    """

    @wraps(func)
    def wrapper(self, x_or_probability=None, *args, **kwargs):
        # if x_or_probability is not given, x_or_probability becomes 1.0
        if x_or_probability is None:
            x_or_probability = 1.0
        if '__no_record' in kwargs and kwargs['__no_record'] is True:
            # pop __no_record
            kwargs.pop('__no_record')
            # no record
            return func(self, x_or_probability, **kwargs)
        # they are recorded
        elif isinstance(x_or_probability, float) or isinstance(x_or_probability, int):
            # probability case
            probability = float(x_or_probability)
            # 0<=np.random.rand()<=1
            if probability == 1.0 or probability >= np.random.rand():
                if self.x is None:
                    if self.info is None:
                        self.info = dict_n({'pc': 0})
                    self.info[self.info['pc']] = {}
                    self.info[(self.info['pc'], 'execute')] = False
                else:
                    # func needs only **kwargs to change behavior
                    self.x, self.info[self.info['pc']] = func(self, self.x, **kwargs)
                    # func has been executed
                    self.info[(self.info['pc'], 'execute')] = True
            else:
                # nothing happened
                self.info[(self.info['pc'], 'execute')] = False
        else:
            # x case
            # check info is given or not
            if not len(args) == 0 and isinstance(args[0], dict_n):
                info = args[0]
            else:
                info = None
            # set x and info
            self(x_or_probability, info=info)
            self.x, self.info[self.info['pc']] = func(self, self.x, **kwargs)
            # func has been executed
            self.info[(self.info['pc'], 'execute')] = True
        # record function name
        self.info[(self.info['pc'], 'whoami')] = func.__name__
        # record default arguments
        # None, None => self, x_or_probability
        tmp = extract_defaults(func)
        for key, val in tmp.items():
            self.info[(self.info['pc'], key)] = val
        # record **kwargs
        for key in kwargs.keys():
            self.info[(self.info['pc'], key)] = kwargs[key]
        # increment program counter
        self.info['pc'] += 1
        return self
    return wrapper


def extract_defaults(func):
    tmp = inspect.getargspec(func)
    if tmp.defaults is None:
        return {}
    return dict(zip(tmp.args[-len(tmp.defaults):], tmp.defaults))


class DataAugmentationPicture(object):

    """Some useful functions for data-augmentation about pictures are defined

    Attributes:
        self.x Optional([numpy.ndarray, str]): input x
        self.info (nutszebra_basic_dictionary.NutszebraDictionary): the info tells about what happened while executing data-augmentation
        eigenvalue (numpy.ndarray): eigenvalue for self.rgb_shift
        eigenvector (numpy.ndarray): eigenvector for self.rgb_shift
        papers (list): the title of papers
        parameters (dict): the parameters in papers are stored
    """

    def __init__(self, **kwargs):
        self.x = None
        self.info = None
        self.eigenvalue = None
        self.eigenvector = None
        self.papers = ['scalable bayesian optimization using deep neural networks',
                       'do deep convolutional nets really need to be deep (or even convolutional)?',
                       ]
        self.parameters = {}
        self.parameters[self.papers[0]] = {'pixel-dropout': {'probability': 0.2},
                                           'shift-hue': {'low': -31.992, 'high': 31.992},
                                           'shift-saturation': {'low': -0.10546, 'high': 0.10546},
                                           'shift-value': {'low': -0.24140, 'high': 0.24140},
                                           'stretch-saturation': {'low': 1. / (1. + 0.31640), 'high': 1. + 0.31640},
                                           'stretch-value': {'low': 1. / (1. + 0.13671), 'high': 1. + 0.13671},
                                           'stretch-BGR': {'low': 1. / (1. + 0.24140), 'high': 1. + 0.24140},
                                           }
        self.parameters[self.papers[1]] = {'shift-hue': {'low': -0.06, 'high': 0.06},
                                           'shift-saturation': {'low': -0.26, 'high': 0.26},
                                           'shift-value': {'low': -0.20, 'high': 0.20},
                                           'stretch-saturation': {'low': 1. / (1. + 0.21), 'high': 1. + 0.21},
                                           'stretch-value': {'low': 1. / (1. + 0.13), 'high': 1. + 0.13},
                                           }

    def __call__(self, x=None, info=None):
        """Set x and info

        Edited date:
            160707

        Test:
            160708

        Example:

        ::

        da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
        da('lenna.jpg')
        >>> print(da.x)
            'lenna.jpg'
        >>> print(da.info)
            {'pc': 0}

        da()
        >>> print(da.x)
            None
        >>> print(da.info)
            None

        da('lenna.jpg', {'test':0})
        >>> print(da.x)
            'lenna.jpg'
        >>> print(da.x)
            {'test':0}

        Args:
            x Optional([None, str, numpy.ndarray, int, float]): If None, self.x and self.info are setted as None, otherwise set this argument as self.x
            info Optional([None, nutszebra_basic_dictionary.NutszebraDictionary, dict]): If None, generate new info, otherwise set this argument as self.info

        Returns:
            nutszebra_data_augmentation_picture.DataAugmentationPicture: return self
        """
        if x is None:
            # reset
            self.x = None
            self.info = None
        else:
            # set
            if info is None:
                info = self.generate_info()
            self.x = x
            self.info = dict_n(info)
        return self

    def end(self, info_flag=False):
        """Return self.x and self.info

        Edited date:
            160707

        Test:
            160708

        Example:

        ::

        da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
        da('lenna.jpg')
        >>> print(da.end())
            'lenna.jpg'
        >>> print(da.end(True))
            ('lenna.jpg', {'pc': 0})

        Args:
            info_flag (bool): If False, only self.x is returned, otherwise return (self.x, self.info).

        Returns:
            numpy.ndarray: info_flag=False
            tuple: info=True, (numpy.ndarray, nutszebra_basic_dictionary.NutszebraDictionary)
        """
        if info_flag is True:
            return (self.x, self.info)
        else:
            return self.x

    @staticmethod
    def generate_info():
        """Generate info dictionary

        Edited date:
            160702

        Test:
            160702

        Example:

        ::

            info = self.nz.generate_info()
            >>> print(info)
                {'pc': 1}

        Returns:
            info (nutszebra_basic_dictionary.NutszebraDictionary): information about what happend while self.execute is written onto this
        """
        return dict_n({'pc': 0})

    def register_eigen(self, data):
        """calculate and register eigenvalue & eigenvector, those eigen ones are used for rgb_shift

        Edited date:
            160422

        Test:
            160501

        Example:

        ::

            self.register_eigen(data)

        Args:
            data (numpy.ndarray): data's ndim has to be 2

        Returns:
            True if successful, False otherwise
        """

        if not data.ndim == 2:
            return False
        cov = np.dot(data, data.T) / len(data)
        V, D, _ = np.linalg.svd(cov)
        self.eigenvalue = D
        self.eigenvector = V.T
        return True

    def _one_rgb_shift(self, mean=0.0, variance=0.1):
        """Private method for self.rgb_shift

        Edited date:
            160501

        Test:
            160501

        Args:
            mean (float): mean for the gaussian distribution
            variance (float): variance for the gaussian distribution

        Returns:
            numpy.ndarray : rgb_shift
        """

        jitter = np.random.normal(mean, variance, self.eigenvalue.shape) * self.eigenvalue
        return self.eigenvector.dot(jitter)

    @staticmethod
    def pick_random_interpolation():
        """The way of interpolation of scipy.misc.imresize

        Edited date:
            160422

        Test:
            160708

        Note:
            | nearest
            | bilinear
            | bicubic
            | cubic

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            answer = da.pick_random_interpolation()
            >>> print(answer)
                bilinear

        Args:

        Returns:
            str: the way of interpolation
        """

        possibility = ['nearest', 'bilinear', 'bicubic', 'cubic']
        index = sampling.pick_random_permutation(1, 4)[0]
        return possibility[index]

    @staticmethod
    def get_keypoints_randomly_for_cropping(picture_shape, sizes):
        """Get keypoints randomly for cropping picture

        Edited date:
            160422

        Test:
            160708

        Example:

        ::

            psudo_picture = np.array(list(six.moves.range(100*100*3))).reshape((100, 100, 3))
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            keypoints = da.get_keypoints_randomly_for_cropping(psudo_picture.shape, (10, 10))
            >>> print(keypoints)
                ((12, 22), (87, 97))

        Args:
            picture_shape (tuple): (height, width)
            sizes (tuple): sizes[0] is height (y) and sizes[1] is width (x)

        Returns:
            tuple: return keypoint, ((start_y, end_y), (start_x, end_x))
        """

        y, x = picture_shape
        length_y, length_x = sizes
        # pick random number
        keypoint_y = sampling.pick_random_permutation(1, y - length_y + 1)[0]
        keypoint_x = sampling.pick_random_permutation(1, x - length_x + 1)[0]
        start_y = keypoint_y
        # end_y does not exceed picture_shape, because number is sampled from y - length_y + 1
        end_y = keypoint_y + length_y
        start_x = keypoint_x
        # end_x does not exceed picture_shape, because number is sampled from x - length_x + 1
        end_x = keypoint_x + length_x
        return ((start_y, end_y), (start_x, end_x))

    @staticmethod
    def crop_picture(picture, keypoints):
        """Get cropped picture according to keypoints

        Edited date:
            160422

        Test:
            160708

        Example:

        ::

            psudo_picture = np.array(list(six.moves.range(10*10*3))).reshape((10, 10, 3))
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            keypoints = da.get_keypoints_randomly_for_cropping(psudo_picture.shape, (3, 3))
            cropped_picture = da.crop_picture(psudo_picture, keypoints)
            >>> print(keypoints)
                ((0, 3), (5, 8))
            >>> print(cropped_picture.shape)
                (3, 3, 3)

            >>> print(cropped_picture)
            array([[[15, 16, 17],
                    [18, 19, 20],
                    [21, 22, 23]],
                   [[45, 46, 47],
                    [48, 49, 50],
                    [51, 52, 53]],
                   [[75, 76, 77],
                    [78, 79, 80],
                    [81, 82, 83]]])

        Args:
            picture (numpy.ndarray): loaded picture
            keypoints (tuple): keypoints ((start_y, end_y), (start_x, end_x)) that is created by self.get_keypoints_randomly

        Returns:
            numpy.ndarray: cropped picture
        """

        start_y, end_y = keypoints[0]
        start_x, end_x = keypoints[1]
        return picture[start_y:end_y, start_x:end_x]

    @staticmethod
    def normalize_pictures_locally(pictures, value=0.):
        """Normalize pictures

        Edited date:
            160515

        Test:
            160711

        Note:
            | The equation for normalization: (x - mean) / sqrt(variance + value)
            | value 0 is typical case and the default value for arguments, but setting value as 10 for a picture normalization is the good choice to suppress noises.
            | You need to flatten pictures before applying normalize_pictures_locally:
            |     (number of pictures, height, width, channel) -> (number of pictures, height * width * channel)

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            pictures = np.random.normal(10., 2., (200, 100 * 100 * 3))
            normalized_pictures = da.normalize_pictures_locally(pictures)
            >>> print(pictures.shape)
                (200, 30000)
            >>> print(normalized_pictures.shape)
                (200, 30000)
            >>> print(np.mean(normalized_pictures[0]))
                -1.1297629498585594e-16
            >>> print(np.var(normalized_pictures))
                1.0

        Args:
            picture (numpy.ndarray): loaded flatten pictures
            value (float): for RGB pictures, value 10 is a good start point. Check at Note.

        Returns:
            numpy.ndarray : the normalized pictures
        """
        number_of_picture, length = len(pictures), len(pictures[0])
        # calculate variance
        # add value here
        var = np.var(pictures, axis=1) + value
        # increase dimension: 1 to 2
        var = np.reshape(var, (len(var), 1))
        var = np.repeat(var, length, axis=1)
        std = np.sqrt(var)
        # calculate mean
        mean = np.mean(pictures, axis=1)
        # increase dimension: 1 to 2
        mean = np.reshape(mean, (len(mean), 1))
        mean = np.repeat(mean, length, axis=1)
        return np.subtract(pictures, mean) / std

    @staticmethod
    def zca_whitening(pictures, regularization=1.0 * 10 ** -5, value=10., normalize_flag=True):
        """Execute zca whitening

        Edited date:
            160515

        Test:
            160515

        Note:
            | You need to flatten pictures before applying zca whitening:
            |     (number of pictures, height, width, channel) -> (number of pictures, height * width * channel)
            | regularization:
            |     zca whitening convert X to V * (D + regularization)^-0.5 * V.T * X
            |     If some eigenvalues are zero, then (D)^-0.5 goes to infinite, thus small number is added to prevent it.

        Example:

        ::

            pictures = np.random.normal(0., 1., (200, 100 * 100 * 3))
            zca_whitening = self.zca_whitening(pictures)
            >>> print(pictures.shape)
                (200, 30000)
            >>> print(zca_whitening.shape)
                (200, 30000)

        Args:
            pictures (numpy.ndarray): loaded flatten pictures
            regularization (float): regularization for zca whitening, please check at Note
            value (float): Check at self.normalize_picture
            normalize_flag (bool): If True, normalization will be executed before zca whitening

        Returns:
            numpy.ndarray : the whitened pictures
        """

        if normalize_flag:
            pictures = DataAugmentationPicture.normalize_pictures_locally(pictures, value=value)
        return DataAugmentationPicture._zca_whitening(np.array(pictures), regularization=regularization)

    @staticmethod
    def _zca_whitening(pictures, regularization=1.0 * 10 ** -6):
        """Private method for zca whitening

        Edited date:
            160515

        Test:
            160515

        Args:
            pictures (numpy.ndarray): loaded flatten pictures
            regularization (float): regularization for zca whitening, please check at self.zca_whitening

        Returns:
            numpy.ndarray : the whitened pictures
        """
        cov = np.dot(pictures, pictures.T) / len(pictures)
        V, D, _ = np.linalg.svd(cov)
        W = np.dot(V, np.diag(1 / np.sqrt(D + regularization)))
        W = np.dot(W, V.T)
        return np.dot(W, pictures)

    @staticmethod
    def find_biggest_picture(pictures):
        """Find the biggest size in pictures

        Edited date:
            160712

        Test:
            160712

        Example:

        ::

            pictures = [np.ones((3, 30, 10)), np.ones((3, 20, 10)), np.ones((3, 10, 20))]
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            answer = da.find_biggest_picture(pictures)
            >>> print(answer)
                (3, 30, 20)

        Args:
            pictures (list): it contains pictures inside

        Returns:
            tuple : the size of the biggest picture
        """

        channel, height, width = (1, 1, 1)
        for picture in pictures:
            tmp_channel, tmp_height, tmp_width = picture.shape
            channel = np.max([channel, tmp_channel])
            height = np.max([height, tmp_height])
            width = np.max([width, tmp_width])
        return (channel, height, width)

    @staticmethod
    def zero_padding(pictures, sizes='biggest', dtype=None):
        """Pad all pictures with zero

        Edited date:
            160712

        Test:
            160712

        Example:

        ::

            pictures = [np.ones((3, 30, 10)), np.ones((3, 20, 10)), np.ones((3, 10, 20))]
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            answer = da.zero_padding(pictures)
            >>> print(answer[0])
                (3, 30, 20)
            >>> print(np.sum(answer[0]))
                900
            >>> print(3 * 30 * 10)
                900
            >>> print(np.all(answer[0][:, 0:30, 0:10] == np.ones((3, 30, 10))))
                True

        Args:
            pictures (list): it contains pictures inside
            size Optional([tuple, 'biggest']): If 'biggest', calculate the biggest size of pictures. You can decide the size by giving tuple(channle, height, width)

        Returns:
            list : zero padded pictures
        """

        if sizes == 'biggest':
            try:
                sizes = DataAugmentationPicture.find_biggest_picture(pictures)
            except:
                return pictures
        if dtype is None:
            dtype = pictures[0].dtype
        answer = []
        for picture in pictures:
            zero = np.zeros(sizes, dtype=dtype)
            channel, height, width = picture.shape
            zero[0:channel, 0:height, 0:width] = picture
            answer.append(zero)
        return answer

    @execute_based_on_probability
    def load_picture(self, x_or_probability, dtype=None, ndim=3, channel=3):
        """Load picture (this is the wrapper of nutszebra.PreprocessPicture.load_picture)

        Edited date:
            160506

        Test:
            160708

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da('lenna.jpg').load_picture()
            >>> print(da.x.shape)
                (855, 400, 3)
            >>> print(da.info)
                {'0': {'dtype': None,
                       'execute': True,
                       'path': 'lenna.jpg',
                       'whoami': 'load_picture'},
                 'pc': 1}

        Args:
            x_or_probability Optional([int, float, str]): If int or float, this argument is considered as the probability and self.x is used for load_picture. If str, set this argument as self.x and execute load_picture with self.x.
            dtype (Optional[None, np.int32, np.float32...]): np.dtype or None
            __no_record (bool): the value of __no_record change the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        try:
            info = {'path': x_or_probability, 'ndim': ndim, 'channel': channel}
            img = preprocess.load_picture(x_or_probability, dtype=dtype)
            if ndim == 2 and img.ndim == ndim:
                return (img, info)
            if ndim == 3 and img.ndim == ndim:
                return (img[:, :, :channel], info)
            if ndim == 3 and img.ndim == 2 and channel == 3:
                return (np.array([img.copy(), img.copy(), img.copy()]).transpose(1, 2, 0), info)
            else:
                return (None, info)
        except:
            return (None, info)

    @execute_based_on_probability
    def horizontal_flipping(self, x_or_probability):
        """Horizontal flipping

        Edited date:
            160422

        Test:
            160708

        Example:

        ::

            pseudo_picture = np.array([np.identity(3), np.identity(3), np.identity(3)])
            >>> print(pseudo_picture[0])
                array([[1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.]])

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da(pseudo_picture).horizontal_flipping()
            >>> print(da.x)
                array([[0., 0., 1.],
                       [0., 1., 0.],
                       [1., 0., 0.]])

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for horizontal_flipping. If str or numpy.ndarray, set this argument as self.x and execute horizontal_flipping with self.x.
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        return (cv2.flip(x_or_probability, 1), {})

    @execute_based_on_probability
    def convert_to_image_format(self, x_or_probability):
        """Convert the picture (channel, height, width) to the picture (height, width, channel)

        Edited date:
            160425

        Test:
            160708

        Example:

        ::

            pseudo_picture = np.ones((3, 32, 32))
            >>> print(pseudo_picture.shape)
                (3, 32, 32)

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da(pseudo_picture).convert_to_image_format()
            >>> print(da.x.shape)
                (32, 32, 3)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute convert_to_image_format with self.x.
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """

        return (np.transpose(x_or_probability, (1, 2, 0)), {})

    @execute_based_on_probability
    def convert_to_chainer_format(self, x_or_probability):
        """Convert the picture (height, width, channel) to the picture (channel, height, width)

        Edited date:
            160425

        Test:
            160708

        Example:

        ::

            pseudo_picture = np.ones((32, 32, 3))
            >>> print(pseudo_picture.shape)
                (32, 32, 3)

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da(pseudo_picture).convert_to_chainer_format()
            >>> print(answer.shape)
                (3, 32, 32)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute convert_to_chainer_format with self.x.
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """

        if x_or_probability.ndim == 2:
            return (x_or_probability[np.newaxis], {})
        return (np.transpose(x_or_probability, (2, 0, 1)), {})

    @execute_based_on_probability
    def resize_image_randomly(self, x_or_probability, size_range=(256, 512), interpolation='random', mode='RGB'):
        """Resize an image randomly

        Edited date:
            160422

        Test:
            160708

        Note:
            | interpolation can be:
            |     nearest
            |     bilinear
            |     bicubic
            |     cubic
            |     random

        Example:

        ::

            pseudo_picture_asymmetry = np.array(np.random.uniform(0, 255, (100, 50, 3)), dtype=np.int)
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.resize_image_randomly(pseudo_picture_asymmetry, size_range=(128, 196), interpolation='random')
            >>> print(da.x.shape)
                (262, 131, 3)
            >>> print(da.info)
                {'0': {'actual_interpolation': 'nearest',
                       'execute': True,
                       'interpolation': 'random',
                       'original_size': (100, 50),
                       'resized_size': (262, 131),
                       'size_range': (128, 196),
                       'whoami': 'resize_image_randomly'},
                'pc': 1}

            da()
            da.resize_image_randomly(pseudo_picture_asymmetry, size_range=(128, 196), interpolation='bicubic')
            >>> print(da.x.shape)
                (340, 170, 3)
            >>> print(da.info)
                {'0': {'actual_interpolation': 'bicubic',
                       'execute': True,
                       'interpolation': 'bicubic',
                       'original_size': (100, 50),
                       'resized_size': (340, 170),
                       'size_range': (128, 196),
                       'whoami': 'resize_image_randomly'},
                'pc': 1}

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute resize_image_randomly with self.x.
            size_range (tuple): the range of size,  such as (256, 512)
            interpolation (str): random means that the way of interpolations will be randomly selected, please check Note
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """

        # randomly select the way of interpolation
        if interpolation == 'random':
            interpolation = DataAugmentationPicture.pick_random_interpolation()
        # if size_range = (256, 512)
        # Pick one random number from the range of 0 and 255 (512 - 256)
        oneside_length = sampling.pick_random_permutation(1, size_range[1] - size_range[0] + 1)[0]
        # Add random number to size_range[0]
        oneside_length = size_range[0] + oneside_length
        y, x = x_or_probability.shape[:2]
        # calculate the size to keep the ratio of the picture
        # find smaller side of picture and then calculate the scale
        if y <= x:
            scale = float(oneside_length) / y
            sizes = (oneside_length, int(scale * x))
        else:
            scale = float(oneside_length) / x
            sizes = (int(scale * y), oneside_length)
        info = {'resized_size': sizes, 'original_size': (y, x), 'actual_interpolation': interpolation, 'scale': scale}
        return (preprocess.resize_image(x_or_probability, sizes, interpolation=interpolation, mode=mode), info)

    @execute_based_on_probability
    def crop_picture_randomly(self, x_or_probability, sizes=(224, 224)):
        """crop picture out randomly with the size

        Edited date:
            160422

        Test:
            160711

        Example:

        ::

            pseudo_picture_asymmetry = np.array(np.random.uniform(0, 255, (100, 50, 3)), dtype=np.int)
            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.crop_picture_randomly(pseudo_picture_asymmetry, sizes=(10, 10))
            >>> print(da.x.shape)
                (10, 10, 3)
            >>> print(da.info)
                {'0': {'execute': True,
                       'keypoints': ((11, 21), (19, 29)),
                       'original_size': (100, 50),
                       'sizes': (10, 10),
                       'whoami': 'crop_picture_randomly'},
                 'pc': 1}

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_picture_randomly with self.x.
            sizes (tuple): crop size, (y, x)
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """

        y, x = x_or_probability.shape[0:2]
        keypoints = DataAugmentationPicture.get_keypoints_randomly_for_cropping((y, x), sizes)
        info = {'keypoints': keypoints, 'original_size': (y, x)}
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), info)

    @execute_based_on_probability
    def rgb_shift(self, x_or_probability, mean=0.0, variance=0.1):
        """Execute rgb_shift

        Edited date:
            160501

        Test:
            160711

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.register_eigen(data)
            rgb_shifted_picture = da.rgb_shift(picture)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute rgb_shift with self.x.
            mean (float): mean for the gaussian distribution
            variance (float): variance for the gaussian distribution
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """

        y, x, channel = x_or_probability.shape
        shifted_pic = np.zeros((y, x, channel), dtype=x_or_probability.dtype)
        element_y = six.moves.range(y)
        element_x = six.moves.range(x)
        for i, ii in itertools.product(element_y, element_x):
            # rgb shift
            shifted_pic[i][ii] = x_or_probability[i][ii] + self._one_rgb_shift(mean=mean, variance=variance)
        return (shifted_pic, {})

    @execute_based_on_probability
    def crop_center(self, x_or_probability, sizes=(384, 384)):
        """Crop the center of the picture

        Edited date:
            160515

        Test:
            160711

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            picture = np.random.normal(0, 0.1, (100, 100, 3))
            da.crop_center(picture, sizes=(10, 10))
            >>> print(da.x.shape)
                (10, 10, 3)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            sizes (tuple): crop size, (y, x)
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        y, x, channel = x_or_probability.shape
        center_y = int(y / 2)
        center_x = int(x / 2)
        frame_y, frame_x = sizes
        up = -int((frame_y + 1) / 2)
        down = int(frame_y / 2)
        left = -int((frame_x + 1) / 2)
        right = int(frame_x / 2)
        start_y = max(center_y + up, 0)
        end_y = min(center_y + down, y)
        start_x = max(center_x + left, 0)
        end_x = min(center_x + right, x)
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def crop_upper_left(self, x_or_probability, sizes=(384, 384)):
        y, x, channel = x_or_probability.shape
        frame_y, frame_x = sizes
        start_y = 0
        end_y = min(frame_y, y)
        start_x = 0
        end_x = min(frame_x, x)
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def crop_top_left(self, x_or_probability, sizes=(384, 384)):
        y, x, channel = x_or_probability.shape
        frame_y, frame_x = sizes
        start_y = 0
        end_y = min(frame_y, y)
        start_x = 0
        end_x = min(frame_x, x)
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def crop_bottom_left(self, x_or_probability, sizes=(384, 384)):
        y, x, channel = x_or_probability.shape
        frame_y, frame_x = sizes
        start_y = max(y - frame_y, 0)
        end_y = y
        start_x = 0
        end_x = min(frame_x, x)
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def crop_top_right(self, x_or_probability, sizes=(384, 384)):
        y, x, channel = x_or_probability.shape
        frame_y, frame_x = sizes
        start_y = 0
        end_y = min(frame_y, y)
        start_x = max(x - frame_x, 0)
        end_x = x
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def crop_bottom_right(self, x_or_probability, sizes=(384, 384)):
        y, x, channel = x_or_probability.shape
        frame_y, frame_x = sizes
        start_y = max(y - frame_y, 0)
        end_y = y
        start_x = max(x - frame_x, 0)
        end_x = x
        keypoints = ((start_y, end_y), (start_x, end_x))
        return (DataAugmentationPicture.crop_picture(x_or_probability, keypoints), {'keypoints': keypoints})

    @execute_based_on_probability
    def subtract_local_mean(self, x_or_probability):
        """Subtract local mean

        Edited date:
            160515

        Test:
            160711

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            path = 'lenna.jpg'
            da.load_picture(path).subtract_local_mean()
            >>> print(da.x)
                -2.1359828785497543e-14
            >>> print(da.info)
                {'0': {'dtype': None,
                       'execute': True,
                       'path': 'lenna.jpg',
                       'whoami': 'load_picture'},
                 '1': {'execute': True,
                       'mean': 95.497653021442488,
                       'whoami': 'subtract_local_mean'},
                 'pc': 2}

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        mean = preprocess.calculate_local_average(x_or_probability)
        return (x_or_probability - mean, {'mean': mean})

    @execute_based_on_probability
    def normalize_picture(self, x_or_probability, value=0., each_rgb=False, dtype=np.float32):
        """Normalize the picture

        Edited date:
            160515

        Test:
            160711

        Note:
            | The equation for normalization: (x - mean) / sqrt(variance + value)
            | value 0 is typical case and the default value for arguments, but setting value as 10 for a picture normalization is the good choice to suppress noises.

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            path = 'lenna.jpg'
            da.load_picture(path).normalize_picture()
            >>> print(np.mean(da.x))
                1.21000026113065e-16
            >>> print(np.var(da.x))
                1.0

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            value (float): for an RGB picture, value 10 is a good start point. Check at Note.
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        x_or_probability = x_or_probability.astype(dtype)
        if each_rgb:
            var = np.var(x_or_probability, axis=(0, 1))
            std = np.sqrt(var + value)
            mean = np.mean(x_or_probability, axis=(0, 1))
            for i in six.moves.range(x_or_probability.shape[2]):
                x_or_probability[:, :, i] = (x_or_probability[:, :, i] - mean[i]) / std[i]
            return (x_or_probability, {'mean': mean, 'var': var, 'std': std})
        else:
            var = np.var(x_or_probability)
            std = np.sqrt(var + value)
            mean = preprocess.calculate_local_average(x_or_probability)
            return ((x_or_probability - mean) / std, {'mean': mean, 'var': var, 'std': std})

    @execute_based_on_probability
    def shift_global_hsv_randomly(self, x_or_probability, hsv='h', low=(-31.992, -0.10546, -0.24140), high=(31.992, 0.10546, 0.24140), ceiling=True):
        """Shift HSV globally

        Edited date:
            160515

        Test:
            160712

        Note:
            | default low and high parameters are from the paper: Scalable Bayesian Optimization Using Deep Neural Networks
            | url: http://arxiv.org/abs/1502.05700
            | For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
            | Different softwares use different scales, thus if you compares values with OpenCV, you need to normalize in the range I mentioned above.

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.load_picture(path).shift_global_hsv_randomly()
            >>> print(da.x.shape)
                (855, 400, 3)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            hsv Optional(['h', 's', 'v']): 'h' is hue, 's' is saturation and 'v' is value
            low (tuple): lower bound of random value of HSV
            high (tuple): higher bound of random value of HSV
            ceiling (bool): If true, exceeded numbers will be fixed
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        # index
        index = 'hsv'.index(hsv)
        # save dtype
        dtype = x_or_probability.dtype
        # convert to HSV
        picture = preprocess.to_hsv(x_or_probability)
        # (height, width, channel) -> (channel, height, width)
        picture, _ = self.convert_to_chainer_format(picture, __no_record=True)
        # convert to float
        picture = np.array(picture, dtype=np.float)
        # generate random value
        actual_low = low[index]
        actual_high = high[index]
        random_hsv = np.random.uniform(low=actual_low, high=actual_high)
        # picture[0] is hue
        # shift
        picture[index] = picture[index] + random_hsv
        # Make the value that is bigger than 179 be 179
        lid = [179, 255, 255]
        if ceiling:
            indices = np.where(picture >= lid[index])
            for i, ii, iii in six.moves.zip(indices[0], indices[1], indices[2]):
                picture[i][ii][iii] = lid[index]
            indices = np.where(picture <= 0)
            for i, ii, iii in six.moves.zip(indices[0], indices[1], indices[2]):
                picture[i][ii][iii] = 0
        # (channel, height, width) -> (height, width, channel)
        picture, _ = self.convert_to_image_format(picture, __no_record=True)
        # convert to BGR
        picture = preprocess.from_hsv(np.array(picture, dtype=np.uint8))
        # get dtype back
        picture = np.array(picture, dtype=dtype)
        return (picture, {'random_hsv': random_hsv, 'actual_low': actual_low, 'actual_high': actual_high})

    @execute_based_on_probability
    def stretch_global_bsv_randomly(self, x_or_probability, bsv='b', low=(1. / (1. + 0.24140), 1. / (1. + 0.31640), 1. / (1. + 0.13671)), high=(1. + 0.24140, 1. + 0.31460, 1. + 0.13671), ceiling=True):
        """Stretch BGR, Saturation or Value globally

        Edited date:
            160515

        Test:
            160712

        Note:
            | default low and high parameters are from the paper: Scalable Bayesian Optimization Using Deep Neural Networks
            | url: http://arxiv.org/abs/1502.05700
            | For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].
            | Different softwares use different scales. So if you are comparing OpenCV values with them, you need to normalize these ranges.

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.load_picture(path).stretch_global_bsv_randomly()
            >>> print(da.x.shape)
                (855, 400, 3)

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            bsv Optional(['b', 's', 'v']): 'b' is BGR, 's' is saturation and 'v' is value
            low (float): lower bound of random value
            high (float): higher bound of random value
            ceiling (bool): If true, exceeded numbers will be fixed
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        np.random.seed()
        # index
        index = 'bsv'.index(bsv)
        # save dtype
        dtype = x_or_probability.dtype
        if bsv is 'b':
            # BGR
            picture = x_or_probability
        else:
            # convert to HSV
            picture = preprocess.to_hsv(x_or_probability)
        # (height, width, channel) -> (channel, height, width)
        picture, _ = self.convert_to_chainer_format(picture, __no_record=True)
        # convert to float
        picture = np.array(picture, dtype=np.float)
        # generate random value
        actual_low = low[index]
        actual_high = high[index]
        random_value = np.random.uniform(low=actual_low, high=actual_high)
        # stretch
        if index is 0:
            # BGR
            picture = picture * random_value
        else:
            # Saturation or Vaue
            picture[index] = picture[index] * random_value
        # Make the value that is bigger than 255 be 255
        if ceiling:
            indices = np.where(picture >= 255)
            for i, ii, iii in six.moves.zip(indices[0], indices[1], indices[2]):
                picture[i][ii][iii] = 255
        # (channel, height, width) -> (height, width, channel)
        picture, _ = self.convert_to_image_format(picture, __no_record=True)
        if index is not 0:
            # convert to BGR
            picture = preprocess.from_hsv(np.array(picture, dtype=np.uint8))
        # get dtype back
        picture = np.array(picture, dtype=dtype)
        return (picture, {'random_hsv': random_value, 'actual_low': actual_low, 'actual_high': actual_high})

    @execute_based_on_probability
    def dropout_picture_randomly(self, x_or_probability, probability=0.2):
        """Dropout picture's pixel

        Edited date:
            160515

        Test:
            160712

        Note:
            | default probability is from the paper: Scalable Bayesian Optimization Using Deep Neural Networks
            | url: http://arxiv.org/abs/1502.05700

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.load_picture(path).dropout_picture_randomly()
            height, width, channel = da.x.shape
            >>> print(height * width * channel)
                1026000
            >>> print(height * width * channel * 0.2)
                205200.0
            >>> print(np.sum(answer == 0))
                205335

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            probability (float): probability of dropping out
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        np.random.seed()
        height, width, channel = x_or_probability.shape
        indices = np.where(np.random.rand(height, width, channel) <= probability)
        answer = x_or_probability.copy()
        answer[indices] = 0
        return (answer, {'howmany': len(indices[0])})

    @execute_based_on_probability
    def dropout_picture_in_the_range_randomly(self, x_or_probability, probability_range=(0., 0.2)):
        """Dropout picture's pixel in the range of some probability

        Edited date:
            160521

        Test:
            160721

        Note:
            | default probability is from the paper: Scalable Bayesian Optimization Using Deep Neural Networks
            | url: http://arxiv.org/abs/1502.05700
            | In the paper, probability is constant

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.load_picture(path).dropout_picture_in_the_range_randomly()
            height, width, channel = da.x.shape
            >>> print(height * width * channel)
                1026000
            >>> print(da.info[('1', 'actual_probability')])
                0.10593729078355862
            >>> print(height * width * channel  *  0.105)
                107730
            >>> print(np.sum(answer == 0))
                108647

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            probability_range (tuple): probability range, (low, high)
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        np.random.seed()
        floor, ceil = probability_range
        probability = np.random.rand() * (ceil - floor) + floor
        height, width, channel = x_or_probability.shape
        indices = np.where(np.random.rand(height, width, channel) <= probability)
        answer = x_or_probability.copy()
        answer[indices] = 0
        return (answer, {'howmany': len(indices[0]), 'actual_probability': probability})

    @execute_based_on_probability
    def rotate_picture_randomly(self, x_or_probability, low=-10.0, high=10.0, reshape=True):
        """Rotate picture

        Edited date:
            160521

        Test:
            160712

        Example:

        ::

            da = nutszebra_data_augmentation_picture.DataAugmentationPicture()
            da.load_picture(path).rotate_picture_randomly()

        Args:
            x_or_probability Optional([int, float, str, numpy.ndarray]): If int or float, this argument is considered as the probability and self.x is used for convert_to_image_format. If str or numpy.ndarray, set this argument as self.x and execute crop_center with self.x.
            low (float): lower bound of random value (degree)
            high (float): higher bound of random value (degree)
            reshape (bool): If True, rorated picture is reshaped
            __no_record (bool): the value of __no_record changes the value to be returned.

        Returns:
            Optional([tuple, class]): If __no_record is False, return self, otherwise return tuple(shaped x, info)
        """
        np.random.seed()
        random_value = np.random.uniform(low, high)
        return (imrotate(x_or_probability, random_value, reshape=True), {'random_value': random_value})

    @execute_based_on_probability
    def to_bgr(self, x_or_probability):
        return (x_or_probability[:, :, ::-1], {})

    @execute_based_on_probability
    def to_rgb(self, x_or_probability):
        return (x_or_probability[:, :, ::-1], {})

    @execute_based_on_probability
    def subtract_constant_mean(self, x_or_probability, rgb=(123.68, 116.779, 103.939)):
        img = np.array(x_or_probability.copy(), dtype=np.float64)
        r, g, b = rgb
        img[:, :, 0] -= r
        img[:, :, 1] -= g
        img[:, :, 2] -= b
        return (img, {})

    @execute_based_on_probability
    def gray_to_rgb(self, x_or_probability):
        if x_or_probability.ndim == 3:
            return (x_or_probability, {})
        elif x_or_probability.ndim == 2:
            img = np.zeros(x_or_probability.shape + (3, ), dtype=x_or_probability.dtype)
            for i in six.moves.range(3):
                img[:, :, i] = x_or_probability.copy()
            return (img, {})

    @execute_based_on_probability
    def scale_to_one(self, x_or_probability, constant=255., dtype=np.float32):
        x_or_probability = x_or_probability.astype(dtype)
        return (x_or_probability / constant, {'constant': constant})

    @execute_based_on_probability
    def fixed_normalization(self, x_or_probability, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), each_rgb=True, dtype=np.float32):
        """
            If you scale picture beforehand by using DataAugmentationPicture.scale_to_one, you can use default values of mean and std.
            URL: https://github.com/facebookresearch/ResNeXt/tree/master/datasets
        """
        x_or_probability = x_or_probability.astype(dtype)
        if each_rgb:
            for i in six.moves.range(len(mean)):
                x_or_probability[:, :, i] = (x_or_probability[:, :, i] - mean[i]) / std[i]
            return (x_or_probability, {'mean': mean, 'var': [s ** 2 for s in std], 'std': std})
        else:
            return ((x_or_probability - mean) / std, {'mean': mean, 'var': std ** 2, 'std': std})

    @execute_based_on_probability
    def fixed_color_normalization(self, x_or_probability, alphastd=0.1, eigval=(0.2175, 0.0188, 0.0045), eigvec=((-0.5675, 0.7192, 0.4009), (-0.5808, -0.0045, -0.8140), (-0.5836, -0.6948, 0.4203)), each_rgb=True, dtype=np.float32):
        """
            URL: https://github.com/facebookresearch/ResNeXt/tree/master/datasets
        """
        np.random.seed()
        x_or_probability = x_or_probability.astype(dtype)
        alpha = np.tile(np.random.normal(0, alphastd, (1, 3)), (3, 1))
        eigval = np.tile(eigval, (3, 1))
        eigvec = np.sum(np.multiply(eigvec, eigval) * alpha, axis=1)
        for i in six.moves.range(len(eigval)):
            x_or_probability[:, :, i] = x_or_probability[:, :, i] + eigvec[i]
        return (x_or_probability, {'eigval': eigval[0], 'alpha': alpha[0]})

    @execute_based_on_probability
    def cutout(self, x_or_probability, sizes=(16, 16), dtype=np.float32):
        x_or_probability = x_or_probability.astype(dtype)
        y, x = x_or_probability.shape[0:2]
        keypoints = DataAugmentationPicture.get_keypoints_randomly_for_cropping((y, x), sizes)
        start_y, end_y = keypoints[0]
        start_x, end_x = keypoints[1]
        x_or_probability[start_y:end_y, start_x:end_x] = 0.0
        return (x_or_probability, {'keypoints': keypoints, 'original_size': (y, x), 'dtype': dtype})

from skimage import io
from skimage import color
import numpy as np
import six
import nutszebra_preprocess
from scipy.misc import imresize
from PIL import Image


class PreprocessPicture(nutszebra_preprocess.Preprocess):

    """Some useful functions for preprocessing pictures are defined

    Attributes:
    """

    def __init__(self):
        super(PreprocessPicture, self).__init__()

    @staticmethod
    def calculate_average(train_x):
        return PreprocessPicture.calculate_global_average(train_x)

    @staticmethod
    def calculate_global_average(train_x):
        return np.sum(train_x, axis=0) / train_x.shape[0]

    @staticmethod
    def calculate_local_average(picture):
        return np.mean(picture)

    @staticmethod
    def to_grey(picture):
        return color.rgb2grey(picture)

    @staticmethod
    def to_hsv(picture):
        return color.rgb2hsv(picture)

    @staticmethod
    def from_hsv(picture):
        return color.hsv2rgb(picture)

    @staticmethod
    def _save_picture(data, path):
        try:
            io.imsave(path, data)
            return True
        except (KeyError, TypeError):
            return False

    @staticmethod
    def save_picture(data, path):
        return PreprocessPicture._save_picture(data, path)

    @staticmethod
    def reduce_size_and_save(array, path, optimize=True, quality=80):
        img = Image.fromarray(array)
        img.save(path, optimize=optimize, quality=quality)

    @staticmethod
    def _load_picture(path):
        try:
            img = io.imread(path)
            return img
        except (OSError, ValueError, IndexError, UnboundLocalError):
            return False

    @staticmethod
    def load_picture(path, dtype=None):
        """Load a single picture

            Args:
                path (str): path to picture
                dtype (numpy.float32, numpy.float64...): numpy.dtype for loaded picture

            Returns:
                numpy.ndarray: image
        """

        if dtype is None:
            return PreprocessPicture._load_picture(path)
        else:
            img = PreprocessPicture._load_picture(path)
            if img is False:
                return False
            else:
                return np.array(img, dtype=dtype)

    @staticmethod
    def load_pictures(paths, dtype=None):
        """Load multiple pictures

            Args:
                path (list): list that contains paths to picture
                dtype (numpy.float32, numpy.float64...): numpy.dtype for loaded picture

            Returns:
                numpy.ndarray: image
        """

        pictures = [0] * len(paths)
        if dtype is None:
            for i, path in enumerate(paths):
                pictures[i] = PreprocessPicture._load_picture(path,  dtype=None)
            return np.array(pictures)
        else:
            for i, path in enumerate(paths):
                pictures[i] = PreprocessPicture._load_picture(path, dtype=dtype)
            return np.array(pictures)

    @staticmethod
    def picture_is_dead_or_alive(path):
        """Check whether picture is dead or alive

            Args:
                path (str): path to picture

            Returns:
                img if successful, False otherwise
        """

        img = PreprocessPicture._load_picture(path)
        if type(img) == np.ndarray:
            # if readable, return picture list
            return img
        else:
            # if unreadable, return False
            return False

    @staticmethod
    def picture_is_too_small(path, sizes=(32, 32)):
        """Check whether picture is too small or not

            Args:
                path (str): path to picture

            Returns:
                False if picture is too small or unreadable, retun list otherwise
        """

        img = PreprocessPicture.picture_is_dead_or_alive(path)
        if img is False:
            # if unreadable, return False
            return False
        if img.ndim is not 3:
            return False
        y, x, channel = img.shape
        if channel is not 3:
            return False
        if y >= sizes[0] and x >= sizes[1]:
            # if big enough, return picture list
            return img
        else:
            # if picture is too small, return False
            return False

    @staticmethod
    def resize_images(images, sizes=(224, 224), interpolation="bilinear", mode='RGB'):
        """Resize images

        Note:
           | interpolation:
           |     nearest
           |     bilinear
           |     bicubic
           |     cubic
           | mode:
           |     'L' (8-bit pixels, black and white)
           |     'P' (8-bit pixels, mapped to any other mode using a color palette)
           |     'RGB' (3x8-bit pixels, true color)
           |     'RGBA' (4x8-bit pixels, true color with transparency mask)
           |     'CMYK' (4x8-bit pixels, color separation)
           |     'YCbCr' (3x8-bit pixels, color video format)
           |     'I' (32-bit signed integer pixels)
           |     'F' (32-bit floating point pixels)

        Example:

        ::

            x = [np.ones((100, 100, 3)), np.zeros((50, 50, 3))]
            answer = self.resize_images(x, sizes(500, 500), interpolation="bilinear")
            >>> print(answer.shape)
            (2, 500, 500, 3)

        Args:
            images (list): it can be numpy
            sizes (tuple): height, width
            interpolation (str): the way of interpolation, check Note
            mode (str): image format, check Note

        Returns:
            list: resized images
        """
        answer = np.zeros((len(images), sizes[0], sizes[1], 3), dtype=np.float32)
        for i in six.moves.range(len(images)):
            answer[i] = PreprocessPicture.resize_image(images[i], sizes=sizes, interpolation=interpolation, mode=mode)
        return answer

    @staticmethod
    def resize_image(image, sizes=(224, 224), interpolation="bilinear", mode='RGB'):
        """Resize image

        Note:
           | interpolation:
           |     nearest
           |     bilinear
           |     bicubic
           |     cubic
           | mode:
           |     'L' (8-bit pixels, black and white)
           |     'P' (8-bit pixels, mapped to any other mode using a color palette)
           |     'RGB' (3x8-bit pixels, true color)
           |     'RGBA' (4x8-bit pixels, true color with transparency mask)
           |     'CMYK' (4x8-bit pixels, color separation)
           |     'YCbCr' (3x8-bit pixels, color video format)
           |     'I' (32-bit signed integer pixels)
           |     'F' (32-bit floating point pixels)

        Example:

        ::

            x = np.ones((100, 100, 3))
            answer = self.resize_image(x, sizes(500, 500), interpolation="bilinear")
            >>> print(answer.shape)
            (500, 500, 3)

        Args:
            images (list): it can be numpy
            sizes (tuple): height, width
            interpolation: the way of interpolation

        Returns:
            list: resized image
        """

        return imresize(image, sizes, interp=interpolation, mode=mode)

    @staticmethod
    def intersection_over_union(kp1, kp2):
        """Calculate intersection over union

        Example:

        ::

            x = np.ones((100, 100, 3))
            answer = self.resize_image(x, sizes(500, 500), interpolation="bilinear")
            >>> print(answer.shape)
            (500, 500, 3)

        Args:
            kp1 (tuple): ((start_y, end_y), (start_x, end_x))
            kp2 (tuple): ((start_y, end_y), (start_x, end_x))

        Returns:
            float: overlapped ratio
        """
        # tuple: keypoint, ((start_y, end_y), (start_x, end_x))
        tmp_y, tmp_x = kp1
        kp1_y_start, kp1_y_end = tmp_y
        kp1_x_start, kp1_x_end = tmp_x
        kp1_area = np.abs(kp1_x_end - kp1_x_start) * np.abs(kp1_y_end - kp1_y_start)
        tmp_y, tmp_x = kp2
        kp2_y_start, kp2_y_end = tmp_y
        kp2_x_start, kp2_x_end = tmp_x
        kp2_area = np.abs(kp2_x_end - kp2_x_start) * np.abs(kp2_y_end - kp2_y_start)
        x_start = np.max([kp1_x_start, kp2_x_start])
        y_start = np.max([kp1_y_start, kp2_y_start])
        x_end = np.min([kp1_x_end, kp2_x_end])
        y_end = np.min([kp1_y_end, kp2_y_end])
        overlapped_area = (np.max([0.0, x_end - x_start])) * (np.max([0.0, y_end - y_start]))
        iou = overlapped_area / (kp1_area + kp2_area - overlapped_area)
        return iou

    @staticmethod
    def cv2_interpolation(num):
        """Each number corresponds to the way of interpolation

        Note:
           | INTER_NEAREST: 0 - a nearest-neighbor interpolation
           | INTER_LINEAR: 1 - a bilinear interpolation (used by default)
           | INTER_CUBIC: 2 - a bicubic interpolation over 4x4 pixel neighborhood
           | INTER_AREA 3 - resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
           | INTER_LANCZOS4: 4 - a Lanczos interpolation over 8x8 pixel neighborhood

        Example:

        ::

            num = cv2.INTER_NEAREST
            >>> print(num)
            0

            self.cv2_interpolation(cv2.INTER_NEAREST)

        Args:
            num (int): int

        Returns:
            str: the name of interpolation
        """
        if num == 0:
            return 'INTER_NEAREST'
        if num == 1:
            return 'INTER_LINEAR'
        if num == 2:
            return 'INTER_CUBIC'
        if num == 3:
            return 'INTER_AREA'
        if num == 4:
            return 'INTER_LANCZOS4'
        return str(num)

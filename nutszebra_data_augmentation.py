import numpy as np
import nutszebra_data_augmentation_picture
from functools import wraps
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()


def reset(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        da()
        return func(self, *args, **kwargs)
    return wrapper


class DataAugmentationCifar10NormalizeSmall(object):

    @staticmethod
    @reset
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(32, 36)).crop_picture_randomly(1.0, sizes=(32, 32)).cutout(0.5, sizes=(16, 16)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(32, 32), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeMiddle(object):

    @staticmethod
    @reset
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(64, 68)).crop_picture_randomly(1.0, sizes=(64, 64)).cutout(0.5, sizes=(32, 32)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(64, 64), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeBig(object):

    @staticmethod
    @reset
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(128, 132)).crop_picture_randomly(1.0, sizes=(128, 128)).cutout(0.5, sizes=(64, 64)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeBigger(object):

    @staticmethod
    @reset
    def train(img):
        da.convert_to_image_format(img).resize_image_randomly(1.0, size_range=(256, 512)).crop_picture_randomly(1.0, sizes=(224, 224)).cutout(0.5, sizes=(112, 112)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.convert_to_image_format(img).resize_image_randomly(1.0, size_range=(384, 384), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeHuge(object):

    @staticmethod
    @reset
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(299, 512)).crop_picture_randomly(1.0, sizes=(299, 299)).cutout(0.5, sizes=(114, 114)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(406, 406), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeSmall(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(32, 36)).crop_picture_randomly(1.0, sizes=(32, 32)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(32, 32), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeMiddle(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(64, 68)).crop_picture_randomly(1.0, sizes=(64, 64)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(64, 64), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeBig(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(129, 132)).crop_picture_randomly(1.0, sizes=(128, 128)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeBigger(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).gray_to_rgb(1.0).resize_image_randomly(1.0, size_range=(256, 512)).crop_picture_randomly(1.0, sizes=(224, 224)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).gray_to_rgb(1.0).resize_image_randomly(1.0, size_range=(384, 384), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeHuge(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(299, 512)).crop_picture_randomly(1.0, sizes=(299, 299)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(406, 406), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DoNothing(object):

    @staticmethod
    @reset
    def train(img):
        return img, None

    @staticmethod
    @reset
    def test(img):
        return img, None


class Ndim(object):

    def __init__(self, ndim=3):
        self.ndim = ndim

    def train(self, img):
        img = np.array(img)
        if not img.ndim == self.ndim:
            diff = self.ndim - img.ndim
            img = np.reshape(img, (1,) * diff + img.shape)
        return img, None

    def test(self, img):
        img = np.array(img)
        if not img.ndim == self.ndim:
            diff = self.ndim - img.ndim
            img = np.reshape(img, (1,) * diff + img.shape)
        return img, None


class DataAugmentationNormalizeBigOneChannel(object):

    @staticmethod
    @reset
    def train(img):
        da.load_picture(img, ndim=2).resize_image_randomly(1.0, size_range=(129, 132)).crop_picture_randomly(1.0, sizes=(128, 128)).normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    @reset
    def test(img):
        da.load_picture(img, ndim=2).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info

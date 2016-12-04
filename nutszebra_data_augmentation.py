import nutszebra_data_augmentation_picture
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()


class DataAugmentationCifar10NormalizeSmall(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(32, 36)).crop_picture_randomly(1.0, sizes=(32, 32)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(32, 32), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeMiddle(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(64, 68)).crop_picture_randomly(1.0, sizes=(64, 64)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(64, 64), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeBig(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(128, 132)).crop_picture_randomly(1.0, sizes=(128, 128)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeBigger(object):

    @staticmethod
    def train(img):
        da.convert_to_image_format(img).resize_image_randomly(1.0, size_range=(256, 512)).crop_picture_randomly(1.0, sizes=(224, 224)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.convert_to_image_format(img).resize_image_randomly(1.0, size_range=(384, 384), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeHuge(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(299, 303)).crop_picture_randomly(1.0, sizes=(299, 299)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(299, 299), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeSmall(object):

    @staticmethod
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(32, 36)).crop_picture_randomly(1.0, sizes=(32, 32)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(32, 32), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeMiddle(object):

    @staticmethod
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(64, 68)).crop_picture_randomly(1.0, sizes=(64, 64)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(64, 64), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeBig(object):

    @staticmethod
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(128, 132)).crop_picture_randomly(1.0, sizes=(128, 128)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeBigger(object):

    @staticmethod
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(256, 512)).crop_picture_randomly(1.0, sizes=(224, 224)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(384, 384), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationNormalizeHuge(object):

    @staticmethod
    def train(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(299, 303)).crop_picture_randomly(1.0, sizes=(299, 299)).normalize_picture(1.0, value=10.).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da.load_picture(img).resize_image_randomly(1.0, size_range=(299, 299), interpolation='bilinear').normalize_picture(1.0, value=10.).convert_to_chainer_format(1.0)
        return da.x, da.info


class DoNothing(object):

    @staticmethod
    def train(img):
        return img, None

    @staticmethod
    def test(img):
        return img, None

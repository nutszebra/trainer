import nutszebra_data_augmentation_picture
da = nutszebra_data_augmentation_picture.DataAugmentationPicture()


class DataAugmentationCifar10NormalizeSmall(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(28, 36)).crop_picture_randomly(1.0, sizes=(26, 26)).normalize_picture(1.0).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(32, 32), interpolation='bilinear').normalize_picture(1.0).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeMiddle(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(60, 68)).crop_picture_randomly(1.0, sizes=(58, 58)).normalize_picture(1.0).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(64, 64), interpolation='bilinear').normalize_picture(1.0).convert_to_chainer_format(1.0)
        return da.x, da.info


class DataAugmentationCifar10NormalizeBig(object):

    @staticmethod
    def train(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(124, 132)).crop_picture_randomly(1.0, sizes=(122, 122)).normalize_picture(1.0).horizontal_flipping(0.5).convert_to_chainer_format(1.0)
        return da.x, da.info

    @staticmethod
    def test(img):
        da(img).convert_to_image_format(1.0).resize_image_randomly(1.0, size_range=(128, 128), interpolation='bilinear').normalize_picture(1.0).convert_to_chainer_format(1.0)
        return da.x, da.info

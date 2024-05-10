import random
import torchvision.transforms as transforms


class RandomAugmentation(object):

    def __init__(self, cfg, interpolation):
        self.cfg = cfg
        self.transform_mapping = {
            "resize_crop": transforms.RandomResizedCrop(
                size=self.cfg.SIZE,
                scale=self.cfg.CROP_SCALE,
                interpolation=interpolation,
            ),
            "rotate": transforms.RandomRotation(degrees=self.cfg.ROTATE_DEGREES),
            "color_jitter": transforms.ColorJitter(
                brightness=self.cfg.COLOR_JITTER_BRIGHTNESS,
                contrast=self.cfg.COLOR_JITTER_CONTRAST,
            ),
            "affine": transforms.RandomAffine(
                degrees=self.cfg.AFFINE_DEGREES,
                translate=self.cfg.AFFINE_TRANSLATE,
                scale=self.cfg.AFFINE_SCALE,
                interpolation=interpolation,
            ),
        }
        self.horizontal_flip = transforms.RandomHorizontalFlip(
            p=self.cfg.HORIZONTAL_FLIP_PROB
        )

    def __call__(self, img):
        chosen_transforms = random.sample(self.transform_mapping.keys(), k=2)

        composed_transform = transforms.Compose(
            [self.transform_mapping[name] for name in chosen_transforms]
        )
        img = composed_transform(img)
        img = self.horizontal_flip(img)

        return img

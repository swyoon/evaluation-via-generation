import logging

from torchvision.transforms import (
    ColorJitter,
    Compose,
    Normalize,
    RandomChoice,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomRotation,
    ToTensor,
)

from augmentations.augmentations import (
    ColorJitterSimCLR,
    GaussianDequantize,
    RandomRotate90,
    ToGray,
    UniformDequantize,
)

logger = logging.getLogger("ptsemseg")


key2aug = {
    "hflip": RandomHorizontalFlip,
    "rotate": RandomRotate90,
    "rcrop": RandomResizedCrop,
    "cjitter": ColorJitterSimCLR,
    "rgray": RandomGrayscale,
    "GaussianDequantize": GaussianDequantize,
    "UniformDequantize": UniformDequantize,
    "togray": ToGray,
    "normalize": Normalize,
    "totensor": ToTensor,
}


def get_composed_augmentations(aug_dict):
    if aug_dict is None:
        print("Using No Augmentations")
        return None

    augmentations = []
    for aug_key, aug_param in aug_dict.items():
        augmentations.append(key2aug[aug_key](**aug_param))
        print("Using {} aug with params {}".format(aug_key, aug_param))
    return Compose(augmentations)

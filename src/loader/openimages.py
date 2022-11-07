import os

import torch
from PIL import Image


class OpenImages_O:
    def __init__(
        self,
        root,
        image_list_file="openimage_o.txt",
        transform=None,
        split="evaluation",
    ):
        """OpenImages Dataset. targeted for OpenImages-O curated from ViM paper"""
        self.root = root
        image_list_file = os.path.join(root, image_list_file)
        with open(image_list_file, "r") as f:
            self.images = [os.path.join(root, "test", line.strip()) for line in f]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, 0

import os
import json
import random
from PIL import Image
import torchvision.transforms.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def get_transforms(img_size):
    def resize(im, msk):
        im = F.resize(im, [img_size, img_size])
        msk = F.resize(msk, [img_size, img_size], interpolation=Image.NEAREST)
        return im, msk

    def random_rot90(im, msk, p=0.5):
        if random.random() < p:
            k = random.randint(1, 3)
            im = im.rotate(90 * k, expand=True)
            msk = msk.rotate(90 * k, expand=True)
            im, msk = F.center_crop(im, (img_size, img_size)), F.center_crop(msk, (img_size, img_size))
        return im, msk

    def random_hflip(im, msk, p=0.5):
        if random.random() < p:
            im = F.hflip(im)
            msk = F.hflip(msk)
        return im, msk

    def random_vflip(im, msk, p=0.5):
        if random.random() < p:
            im = F.vflip(im)
            msk = F.vflip(msk)
        return im, msk

    # ToTensor and Normalize only apply to image
    image_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    mask_to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0).long())  # Remove channel dim and convert to long
    ])

    def train_transform(image, mask):
        image, mask = resize(image, mask)
        image, mask = random_rot90(image, mask)
        image, mask = random_hflip(image, mask)
        image, mask = random_vflip(image, mask)
        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)
        return image, mask

    def val_transform(image, mask):
        image, mask = resize(image, mask)
        image = image_to_tensor(image)
        mask = mask_to_tensor(mask)
        return image, mask

    return train_transform, val_transform
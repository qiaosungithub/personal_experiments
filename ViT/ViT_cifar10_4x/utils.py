from PIL import Image
import os
import numpy as np
import pickle
from typing import Any, Callable, Optional, Tuple
import math
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
import torchvision
from PIL import Image
import time
from torchvision.datasets.vision import VisionDataset


class CIFAR10_4x(VisionDataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        split : tarin, valid or test
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    base_folder = 'cifar10_4x'

    file_dic = {"train": "train", "valid": "valid", "test": "test"}

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
    }

    def __init__(
        self,
        root: str,
        split: str = 'train',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:

        super(CIFAR10_4x, self).__init__(root)

        self.split = split  # training set or test set
        self.transform = transform

        file_name = self.file_dic[split]

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays

        file_path = os.path.join(self.root, self.base_folder, file_name)
        with open(file_path, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
            self.data.append(entry['data'])
            if 'labels' in entry:
                self.targets.extend(entry['labels'])
            else:
                self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data)  # HWC

        self._load_meta()

    def _load_meta(self) -> None:
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])

        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format(self.split)
    
    data_root_dir = '.'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                         [60 / 255, 59 / 255, 64 / 255])
])

augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(128, padding=6),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([125 / 255, 124 / 255, 115 / 255],
                         [60 / 255, 59 / 255, 64 / 255])
])


def load_cifar10_4x(dir='.', train_batch_size=128, valid_batch_size=256, augment=False):
    validset = CIFAR10_4x(root=dir,
                        split='valid', transform=transform)
    validloader = torch.utils.data.DataLoader(
        validset, batch_size=valid_batch_size, shuffle=False, num_workers=8)
    if augment:
        trainset = CIFAR10_4x(root=dir,
                            split="train", transform=augment_transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
        return trainloader, validloader
    else:

        trainset = CIFAR10_4x(root=dir,
                            split="train", transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=train_batch_size, shuffle=True, num_workers=8)
    
        return trainloader, validloader
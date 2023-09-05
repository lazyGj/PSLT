# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json
import cv2
from torch.functional import split
import torch
import random
import numpy as np
from PIL import Image
import pdb
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import pathlib
# import requests
# import tarfile
# import shutil

# from tqdm import tqdm

V2_DATASET_SIZE = 10000
URLS = {"matched-frequency" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-matched-frequency.tar.gz",
        "threshold-0.7" : "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-threshold0.7.tar.gz",
        "top-images": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenetv2-top-images.tar.gz",
        "val": "https://imagenetv2public.s3-us-west-2.amazonaws.com/imagenet_validation.tar.gz"}

FNAMES = {"matched-frequency" : "imagenetv2-matched-frequency-format-val",
        "threshold-0.7" : "imagenetv2-threshold0.7-format-val",
        "top-images": "imagenetv2-top-images-format-val",
        "val": "imagenet_validation"}

class ImageNetV2Dataset(Dataset):
    def __init__(self, variant="matched-frequency-format-val", transform=None, location="."):
        self.dataset_root = pathlib.Path(f"{location}/imagenetv2-{variant}/")
        self.tar_root = pathlib.Path(f"{location}/imagenetv2-{variant}.tar.gz")
        self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        self.transform = transform
        # assert variant in URLS, f"unknown V2 Variant: {variant}"
        if not self.dataset_root.exists() or len(self.fnames) != V2_DATASET_SIZE:
            print('-------------- dataset errors! -----------')
            # if not self.tar_root.exists():
            #     print(f"Dataset {variant} not found on disk, downloading....")
            #     response = requests.get(URLS[variant], stream=True)
            #     total_size_in_bytes= int(response.headers.get('content-length', 0))
            #     block_size = 1024 #1 Kibibyte
            #     progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            #     with open(self.tar_root, 'wb') as f:
            #         for data in response.iter_content(block_size):
            #             progress_bar.update(len(data))
            #             f.write(data)
            #     progress_bar.close()
            #     if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            #         assert False, f"Downloading from {URLS[variant]} failed"
            # print("Extracting....")
            # tarfile.open(self.tar_root).extractall(f"{location}")
            # shutil.move(f"{location}/{FNAMES[variant]}", self.dataset_root)
            # self.fnames = list(self.dataset_root.glob("**/*.jpeg"))
        

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, i):
        img, label = Image.open(self.fnames[i]), int(self.fnames[i].parent.name)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args, visual=False):
    transform = build_transform(is_train, args)
    visual_transform = build_transform(is_train, args, visual=True) if visual else None

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'ILSVRC2012_train' if is_train else 'ILSVRC2012_val_pytorch')
        print(f'----------------------------{args.data_path}----------------')
        if args.data_path == '/home/data/junkai/imagenet/train_val':
            root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'IMNETV2':
        dataset = ImageNetV2Dataset(variant="matched-frequency-format-val", transform=transform, location=args.data_path)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'IMNET100':
        nb_classes = 100
        label_path = '/home/gaojie/code/code/transformer/deit-main/class.txt'
        image_path = '/home/gaojie/code/code/transformer/deit-main/image_path.txt'
        dataset = imgnet100_dataset(label_path, image_path, train=is_train, transform=transform, visual=visual, visual_transform=visual_transform)

    return dataset, nb_classes


class imgnet100_dataset(Dataset):
    def __init__(self, label_path, image_path, train=True, transform=None, visual=False, visual_transform=None):
        self.class2label = {}
        with open(label_path, 'r') as f:
            class_names = f.readlines()
            for i, class_name in enumerate(class_names):
                self.class2label[class_name.strip('\n')] = i
        
        with open(image_path, 'r') as f:
            image_paths = f.readlines()
        
        random.seed(1)
        random.shuffle(image_paths)
        random.seed()
        num_train = (len(image_paths) // 10) * 9
        if train:
            self.image_paths = image_paths[:num_train]
        elif not visual:
            self.image_paths = image_paths[num_train:]
        else:
            self.image_paths = image_paths[num_train:num_train+1024]
        self.len = len(self.image_paths)
        self.transform = transform
        self.visual_transform = visual_transform

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        img_path = self.image_paths[index].strip('\n')
        img_ori = Image.open(img_path)
        if img_ori.mode != 'RGB':
                img_ori = img_ori.convert('RGB')
        # img = np.transpose(img, [2, 0, 1])
        # pdb.set_trace()
        if self.transform:
            # print(img_path)
            img = self.transform(img_ori)
        if self.visual_transform:
            img_visual = self.visual_transform(img_ori)
        class_name = img_path.split('/')[-2]
        label = self.class2label[class_name]
        # print('finish transforms!!!!')
        if self.visual_transform:
            return img, label, img_visual
        else:
            return img, label


def build_transform(is_train, args, visual=False):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if not visual:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

if __name__ == "__main__":
    label_path = '/home/gaojie/code/transformer/deit-main/class.txt'
    image_path = '/home/gaojie/code/transformer/deit-main/image_path.txt'
    dataset = imgnet100_dataset(image_path, image_path, train=True, transform=None)
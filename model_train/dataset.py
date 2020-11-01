import random

import albumentations as albu
import cv2
import numpy as np
import pandas as pd
import torch
from augmentations import get_training_augmentation, get_validation_augmentation
from constants import DOCNAME2CLASS


class DocumentDatasetAddClsBig:

    def __init__(self,
                 df,
                 augmentation=None,
                 mode='train',
                 use_resized_images=False,
                 ):
        self.df = df
        self.augmentation = augmentation
        self.mode = mode
        self.use_resized_images = use_resized_images

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        #im_path, ann_path = sample['image'], sample['annotation']
        im_path = sample['full_path']
        titul = sample['titul']
        cls_target = DOCNAME2CLASS[sample['class']]

        filename = sample['filename']

        if self.use_resized_images and cls_target != -1:
            image = cv2.imread(im_path.replace('pdf2image','Dataset_resize512'))
        else:
            image = cv2.imread(im_path)

        origin_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = origin_image


        h, w, _, = image.shape
        if h >= w:
            if self.mode == 'train':
                image = albu.RandomCrop(h-random.randint(0,30), w-random.randint(0,30), p=0.5)(image=image)['image']

        image = self.augmentation(image)
        #image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        y_oh = np.zeros(len(DOCNAME2CLASS))
        titul_oh = np.zeros(2)
        titul_oh[titul] = 1
        y_oh[cls_target] = 1
        return image, torch.FloatTensor(y_oh), filename, titul_oh

    def __len__(self):
        return len(self.df)


def get_train_val_datasets_cls_big(train_dataset_path, val_dataset_path, test_dataset_path):
    df_train = pd.read_csv(train_dataset_path)
    df_val = pd.read_csv(val_dataset_path)
    df_test = pd.read_csv(test_dataset_path).sort_values('filename')

    train_dataset = DocumentDatasetAddClsBig(
        df_train,
        augmentation=get_training_augmentation(),
        mode='train',
        use_resized_images=True,
    )

    valid_dataset = DocumentDatasetAddClsBig(
        df_val,
        augmentation=get_validation_augmentation(),
        mode='valid',
        use_resized_images=True,
    )

    test_dataset_path = DocumentDatasetAddClsBig(
        df_test,
        augmentation=get_validation_augmentation(),
        mode='valid',
        use_resized_images=True,
    )

    return train_dataset, valid_dataset, test_dataset_path

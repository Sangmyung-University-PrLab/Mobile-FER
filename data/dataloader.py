import os
import data.CVTransforms as cvTransforms
import data.PILTransform as pilTransforms
import numpy as np

from torch.utils import data
from torchvision import datasets
import torchvision.transforms as transforms
import data.DataSet as myDataLoader
import torch
import data.loadData as ld
import data.load_skin_data as sk_ld
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2


def CVdataloader(data_dir, classes, batch_size, scaleIn, w=180, h=320, num_work=4, test=False):

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.02, rotate_limit=(1,10), p=0.5),
            A.ColorJitter(),
            A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Blur(p=0.1),
            A.CLAHE(p=0.1),
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    np.random.seed(31)

    if test:
        dataset = np.load("test_data.npy", allow_pickle=True)
        np.random.shuffle(dataset)

        print(" Load Affectnet test dataset")
        testLoader = torch.utils.data.DataLoader(
            myDataLoader.CVDataset(dataset, transform=val_transform, data_dir=data_dir),
            batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

        return testLoader
    else:
        dataset = np.load("train_data.npy", allow_pickle=True)
        np.random.shuffle(dataset)

        trainset = dataset[:-round(len(dataset)*0.05)]
        valset = dataset[-round(len(dataset)*0.05):]

        # trainset = dataset[:128]
        # valset = dataset[:128]
        print(" Load Affectnet train dataset")
        trainLoader = torch.utils.data.DataLoader(
            myDataLoader.CVDataset(trainset, transform=train_transform, data_dir=data_dir),
            batch_size=batch_size, shuffle=True, num_workers=num_work, pin_memory=True)

        print(" Load Affectnet val dataset")
        valLoader = torch.utils.data.DataLoader(
            myDataLoader.CVDataset(valset, transform=val_transform, data_dir=data_dir),
            batch_size=batch_size, shuffle=False, num_workers=num_work, pin_memory=True)

        return trainLoader, valLoader


def get_dataloader(args): #cash, dataset_name, data_dir, classes, batch_size, scaleIn=1, w=180, h=320, Edge= False):

    data_dir = "./DB/"
    classes = args["classes"]
    batch_size = args["batch_size"]
    w = args["w"]
    h = args["h"]
    num_work = args["num_work"]
    scaleIn = args["scaleIn"]
    print(" This data load w = %d h = %d scaleIn = %d" % (w, h, scaleIn))
    return CVdataloader(data_dir, classes, batch_size, scaleIn, w=w, h=h, num_work=num_work, test=False)


def get_test_loader(args):

    data_dir = "./DB/"
    classes = args["classes"]
    batch_size = args["batch_size"]
    w = args["w"]
    h = args["h"]
    num_work = args["num_work"]
    scaleIn = args["scaleIn"]
    print(" This data load w = %d h = %d scaleIn = %d" % (w, h, scaleIn))
    return CVdataloader(data_dir, classes, batch_size, scaleIn, w=w, h=h, num_work=num_work, test=True)
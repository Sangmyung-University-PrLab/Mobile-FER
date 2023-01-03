import torch
import cv2
import torch.utils.data
import numpy as np
from PIL import Image


class CVDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, data_dir=None):
        self.dataset = dataset
        self.transform = transform
        self.data_dir = data_dir
        print("This num of data is " +str(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        filepath, expression, valence, arousal, d_neutral, d_happy, d_sadness,\
            d_surprise, d_fear, d_disgust, d_anger, d_contempt, d_valence, d_arousal = self.dataset[idx]
        
        image = cv2.imread(self.data_dir + filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = np.array([int(expression), float(valence), float(arousal)], dtype=np.float32)
        distil = np.array([float(d_neutral), float(d_happy), float(d_sadness), float(d_surprise),float(d_fear), \
            float(d_disgust), float(d_anger), float(d_contempt), float(d_valence), float(d_arousal)], dtype=np.float32)
        
        label = torch.from_numpy(label)
        distil = torch.from_numpy(distil)
        
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = image.astype(np.float32)
            image[0] = (image[0] - 0.485) / 0.229
            image[1] = (image[1] - 0.456) / 0.224
            image[2] = (image[2] - 0.406) / 0.225
            image = torch.from_numpy(image)

        return (image, label, distil)



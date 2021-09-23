import torch
import numpy as np
import cv2

def load_folder(val):
    data = np.load('Datatemp.npz', allow_pickle=True)
    if val == 0:
        images = data['train_x']
        labels = data['train_y']
    if val == 1:
        images = data['val_x']
        labels = data['val_y']
    if val == 2:
        images = data['test_x']
        labels = data['test_y']

    return images, labels


class Dataset():
    def __init__(self, val, size=64, transform=None):
        self.data, self.labels = load_folder(val)
        self.height = size
        self.width = size
        self.transform = transform

    def __getitem__(self, index):
        pixels = cv2.resize(self.data[index], dsize=(self.height, self.width))
        pixels = (torch.from_numpy(pixels).view(-1, self.height, self.width)).float()
        return pixels, self.labels[index]

    def __len__(self):
        return len(self.labels)



def create_SEM_train_dataset(size=64):

    dataset = Dataset(0,size)
    return dataset



def create_SEM_val_dataset(size=64):

    dataset = Dataset(1, size)
    return dataset



def create_SEM_test_dataset(size=64):

    dataset = Dataset(2, size)
    return dataset

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

import os

import pickle

import torchvision.transforms as transforms


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


class CIFAR10(Dataset):

    def __init__(self, root, train, transform=None):
        self.dir = os.path.join(root, 'cifar-10-batches-py/')
        self.train = train
        self.transform = transform

        self.data = {b'labels': [], b'data': []}

        if train:
            for i in range(5):
                batch_name = 'data_batch_' + str(i + 1)
                data = unpickle(os.path.join(self.dir, batch_name))
                self.update_data(data)
        else:
            data = unpickle(os.path.join(self.dir, 'test_batch'))
            self.update_data(data)

        # print(self.data[b'labels'])
        # print(self.data)

        self.reshape_data()

        self.apply_transform()



    def __len__(self):
        return len(self.data[b'labels'])

    def __getitem__(self, index):
        return (self.data[b'data'][index],
                self.data[b'labels'][index])

    def update_data(self, data):
        for key in self.data.keys():
            self.data[key].extend(data[key])

    def apply_transform(self):
        for index, img in enumerate(self.data[b'data']):
            self.data[b'data'][index] = self.transform(img)

    def reshape_data(self):
        for index, array in enumerate(self.data[b'data']):
            img = np.reshape(array, (-1, 3), order='F')
            img = np.reshape(img, (32, 32, 3))
            self.data[b'data'][index] = img



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    dataset = CIFAR10(root='./data', train=False, transform=transform)
    print(len(dataset))
    # print(dataset.__getitem__(0))
    print(dataset[0])
import cv2
import numpy as np

from torchvision import datasets

train_dataset = datasets.MNIST('./data/', train=True, download=True)
print('# MNIST', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000

# train_dataset = datasets.KMNIST('./data/', train=True, download=True)
# print('# KMNIST', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000

# train_dataset = datasets.FashionMNIST('./data/', train=True, download=True)
# print('# FashionMNIST', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000

# train_dataset = datasets.CIFAR10('./data/', train=True, download=True)
# print('# CIFAR-10', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000

# train_dataset = datasets.CIFAR100('./data/', train=True, download=True)
# print('# CIFAR-100', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000

train_dataset = datasets.SVHN('./data/SVHN/', 'train', download=True)
# train_dataset = datasets.SVHN('./data/SVHN/', 'test', download=True)
# train_dataset = datasets.SVHN('./data/SVHN/', 'extra', download=True)
print('# SVHN', len(train_dataset), np.shape(train_dataset[0][0]),  10) # 73,257, 26,032, 531,131

# train_dataset = datasets.STL10('./data/', 'train', download=True)
# train_dataset = datasets.STL10('./data/', 'test', download=True)
# train_dataset = datasets.STL10('./data/', 'unlabeled', download=True)
# print('# STL-10', len(train_dataset), np.shape(train_dataset[0][0]), len(train_dataset.classes)) # 60,000


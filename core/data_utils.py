
from torchvision import datasets
from torchvision import transforms

def get_CIFAR_10(
    data_dir, 
    image_size=32, 
    mean=(0.4914009, 0.48215914, 0.44653103), std=(0.20230275, 0.1994131, 0.2009607),
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_transforms)
    validation_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=test_transforms)

    in_channels = 3
    classes = 10

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_CIFAR_100(
    data_dir, 
    image_size=32, 
    mean=(0.5070757, 0.4865504, 0.44091937), std=(0.20089693, 0.19844234, 0.20229684), 
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transforms)
    validation_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transforms)

    in_channels = 3
    classes = 100

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_STL_10(
    data_dir, 
    image_size=32, 
    mean=(0.44671088, 0.43981022, 0.406646), std=(0.22415751, 0.22150059, 0.22391169), 
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.STL10(data_dir, 'train', download=True, transform=train_transforms)
    validation_dataset = datasets.STL10(data_dir, 'train', download=True, transform=test_transforms)
    test_dataset = datasets.STL10(data_dir, 'test', download=True, transform=test_transforms)

    in_channels = 3
    classes = 10

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_MNIST(
    data_dir, 
    image_size=28, 
    mean=(0.13066047,), std=(0.30150425,), 
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=train_transforms)
    validation_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=test_transforms)

    in_channels = 1
    classes = 10

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_KMNIST(
    data_dir, 
    image_size=28, 
    mean=(0.19176215,), std=(0.33852664,), 
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.KMNIST(data_dir, train=True, download=True, transform=train_transforms)
    validation_dataset = datasets.KMNIST(data_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.KMNIST(data_dir, train=False, download=True, transform=test_transforms)

    in_channels = 1
    classes = 10

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_FashionMNIST(
    data_dir, 
    image_size=28, 
    mean=(0.28604063,), std=(0.32045338,), 
    train_transforms=None, test_transforms=None
):
    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transforms)
    validation_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=test_transforms)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transforms)

    in_channels = 1
    classes = 10

    return train_dataset, validation_dataset, test_dataset, in_channels, classes

def get_SVHN(
    data_dir, 
    image_size=32, 
    mean=(0.43768454, 0.44376868, 0.472804), std=(0.12008653, 0.123137444, 0.10520427), 
    train_transforms=None, test_transforms=None
):
    data_dir = data_dir + 'SVHN/'

    if train_transforms is None:
        train_transforms = transforms.Compose([
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2)),

            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    if test_transforms is None:
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    train_dataset = datasets.SVHN(data_dir, 'train', download=True, transform=train_transforms)
    validation_dataset = datasets.SVHN(data_dir, 'train', download=True, transform=test_transforms)
    test_dataset = datasets.SVHN(data_dir, 'train', download=True, transform=test_transforms)

    in_channels = 3
    classes = 10
    
    return train_dataset, validation_dataset, test_dataset, in_channels, classes

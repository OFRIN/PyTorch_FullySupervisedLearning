

1. CIFAR-10
```
python3 train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name CIFAR-10_seed@0 --dataset_name CIFAR-10
python3 train.py --use_gpu 1 --seed 1 --use_cores 4 --experiment_name CIFAR-10_seed@1 --dataset_name CIFAR-10
python3 train.py --use_gpu 2 --seed 2 --use_cores 4 --experiment_name CIFAR-10_seed@2 --dataset_name CIFAR-10
python3 train.py --use_gpu 3 --seed 3 --use_cores 4 --experiment_name CIFAR-10_seed@3 --dataset_name CIFAR-10
```

2. CIFAR-100
```
python3 train.py --use_gpu 2 --seed 0 --use_cores 4 --experiment_name CIFAR-100_seed@0 --dataset_name CIFAR-100
python3 train.py --use_gpu 2 --seed 1 --use_cores 4 --experiment_name CIFAR-100_seed@1 --dataset_name CIFAR-100
python3 train.py --use_gpu 2 --seed 2 --use_cores 4 --experiment_name CIFAR-100_seed@2 --dataset_name CIFAR-100
python3 train.py --use_gpu 2 --seed 3 --use_cores 4 --experiment_name CIFAR-100_seed@3 --dataset_name CIFAR-100
```

3. STL-10
```
python3 train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name STL-10_seed@0 --dataset_name STL-10
python3 train.py --use_gpu 1 --seed 1 --use_cores 4 --experiment_name STL-10_seed@1 --dataset_name STL-10
python3 train.py --use_gpu 2 --seed 2 --use_cores 4 --experiment_name STL-10_seed@2 --dataset_name STL-10
python3 train.py --use_gpu 3 --seed 3 --use_cores 4 --experiment_name STL-10_seed@3 --dataset_name STL-10
```

4. MNIST
```
python train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name MNIST_seed@0 --dataset_name MNIST --image_size 28
python train.py --use_gpu 0 --seed 1 --use_cores 4 --experiment_name MNIST_seed@1 --dataset_name MNIST --image_size 28
python train.py --use_gpu 0 --seed 2 --use_cores 4 --experiment_name MNIST_seed@2 --dataset_name MNIST --image_size 28
python train.py --use_gpu 0 --seed 3 --use_cores 4 --experiment_name MNIST_seed@3 --dataset_name MNIST --image_size 28
```

5. KMNIST
```
python train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name KMNIST_seed@0 --dataset_name KMNIST --image_size 28
python train.py --use_gpu 0 --seed 1 --use_cores 4 --experiment_name KMNIST_seed@1 --dataset_name KMNIST --image_size 28
python train.py --use_gpu 0 --seed 2 --use_cores 4 --experiment_name KMNIST_seed@2 --dataset_name KMNIST --image_size 28
python train.py --use_gpu 0 --seed 3 --use_cores 4 --experiment_name KMNIST_seed@3 --dataset_name KMNIST --image_size 28
```

6. FashionMNIST 
```
python train.py --use_gpu 0 --seed 0 --use_cores 4 --experiment_name FashionMNIST_seed@0 --dataset_name FashionMNIST --image_size 28
python train.py --use_gpu 0 --seed 1 --use_cores 4 --experiment_name FashionMNIST_seed@1 --dataset_name FashionMNIST --image_size 28
python train.py --use_gpu 0 --seed 2 --use_cores 4 --experiment_name FashionMNIST_seed@2 --dataset_name FashionMNIST --image_size 28
python train.py --use_gpu 0 --seed 3 --use_cores 4 --experiment_name FashionMNIST_seed@3 --dataset_name FashionMNIST --image_size 28
```

7. SVHN
```
python3 train.py --use_gpu 2 --seed 0 --use_cores 4 --experiment_name SVHN_seed@0 --dataset_name SVHN
python3 train.py --use_gpu 2 --seed 1 --use_cores 4 --experiment_name SVHN_seed@1 --dataset_name SVHN
python3 train.py --use_gpu 1 --seed 2 --use_cores 4 --experiment_name SVHN_seed@2 --dataset_name SVHN
python3 train.py --use_gpu 1 --seed 3 --use_cores 4 --experiment_name SVHN_seed@3 --dataset_name SVHN
```
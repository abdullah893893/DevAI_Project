# src/data_loader.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_class_names():
    return ['uçak', 'araba', 'kuş', 'kedi', 'geyik',
            'köpek', 'kurbağa', 'at', 'gemi', 'kamyon']

def load_cifar10(batch_size=64, num_workers=2):
    """
    CIFAR-10 PyTorch DataLoader döndürür.
    """
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

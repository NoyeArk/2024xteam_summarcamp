from torchvision import datasets

train_dataset = datasets.cifar.CIFAR100(root='cifar100', train=True, transform=None, download=True)
test_dataset = datasets.cifar.CIFAR100(root='cifar100', train=False, transform=None, download=True)

print(train_dataset)
print(test_dataset)

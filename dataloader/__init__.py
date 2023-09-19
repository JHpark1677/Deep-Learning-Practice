import torch
import torchvision
import torchvision.transforms as transforms

def dataloader(path, dataset, batch_size):

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
    
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                            download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    if dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=path, train=True,
                                                download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=path, train=False,
                                            download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    return trainloader, testloader
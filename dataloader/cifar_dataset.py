import torch
import torchvision
import torchvision.transforms as transforms

def dataloader(path, dataset, batch_size):

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     ])

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 

    
    if dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                            download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    if dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=path, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=path, train=False,
                                            download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        
    return trainloader, testloader
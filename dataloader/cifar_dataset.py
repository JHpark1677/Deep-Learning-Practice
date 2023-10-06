import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.distributed import DistributedSampler

def dataloader(args):

    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 

    transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]) 

    
    if args.dataset == "cifar10":
        trainset = torchvision.datasets.CIFAR10(root=args.path, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=args.path, train=False,
                                            download=True, transform=transform_test)
        if args.distributed:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=int(args.batch_size/args.world_size), 
                                                      num_workers=int(args.num_workers/args.world_size), 
                                                      pin_memory=True, sampler=DistributedSampler(dataset=trainset, shuffle=True), drop_last=True)
            testloader = torch.utils.data.DataLoader(testset,batch_size=int(args.batch_size/args.world_size),
                                                    num_workers=int(args.num_workers/args.world_size),
                                                    pin_memory=True, sampler=DistributedSampler(dataset=testset, shuffle=False), drop_last=False)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                    shuffle=True, num_workers=2)
            testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=False, num_workers=2)
        
    if args.dataset == "cifar100":
        trainset = torchvision.datasets.CIFAR100(root=args.path, train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=args.path, train=False,
                                            download=True, transform=transform_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                shuffle=True, num_workers=2)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                shuffle=False, num_workers=2)
        
    return trainloader, testloader
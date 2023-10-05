import os
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import visdom

import models_
from dataloader import cifar_dataset
from tools import train_tool
from tools import eval_tool

from utils import init_for_distributed

def main_worker(rank, args):

    if args.distributed:
        init_for_distributed(rank, args)

    # 2. device
    device = torch.device('cuda:{}'.format(int(args.gpu_ids[args.rank])))
    print("what's device? :", device)

    # 3. visdom
    if args.visdom_true:
        vis = visdom.Visdom()
    else:
        vis = None
    # 4. dataset
    
    trainloader, testloader = cifar_dataset.dataloader(args)

    # 5. model
    #model = models_.EfficientNetB0().to(device)
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT').to(device)
    #model = models_.Wide_ResNet(depth=28, widen_factor=10, num_classes=10).to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(module=model, device_ids=[int(args.gpu_ids[args.rank])], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    
    #optimizer = optim.Adadelta(model.parameters())
    optimizer = optim.SGD(param_dicts, lr=0.0001, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.CrossEntropyLoss()
    args.epoch_num = 300
    test_accuracy = 0

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        path = '../checkpoint/' + os.path.join(args.load_ckp)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_acc = 0

    for epoch in range(start_epoch+1, start_epoch+args.epoch_num+1):
        
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
            
        train_tool.train(model, trainloader, optimizer, criterion, epoch, device, vis, args)
        
        test_loss, test_accuracy = eval_tool.evaluate(model, testloader, criterion, device, vis, args)
        scheduler.step()
        if test_accuracy > best_acc :
            print('Saving..')
            state = {
                'model' : model.state_dict(),
                'optimizer' : optimizer.state_dict(), 
                'acc' : test_accuracy,
                'epoch' : epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            path = '../checkpoint/' + os.path.join(args.save_ckp)
            torch.save(state, path)
            best_acc = test_accuracy

        print("\n[EPOCH: {}], \tModel: ResNet, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f}, \tLearning Rate : {:.8f} % \n".format(epoch, 
                                                                                 test_loss, test_accuracy, optimizer.param_groups[0]["lr"]))


if __name__ == "__main__": 
    import torch.multiprocessing as mp
    from config import get_args_parser
    import configargparse

    parser = configargparse.ArgumentParser('ResNet', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if len(args.gpu_ids) > 1:
        args.distributed = True
    else:
        args.distributed = False

    args.world_size = len(args.gpu_ids)
    args.num_workers = len(args.gpu_ids) * 4
    args.distributed = False

    if args.distributed:
        mp.spawn(main_worker,
                 args=(args,),
                 nprocs=args.world_size,
                 join=True)
    else:
        main_worker(args.rank, args)
import os
import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import models_
from dataloader import cifar_dataset
from tools import train_tool
from tools import eval_tool

if __name__ == "__main__": 
   
    from config import get_args_parser
    import configargparse

    parser = configargparse.ArgumentParser('ResNet', parents=[get_args_parser()])
    args = parser.parse_args()

    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = cifar_dataset.dataloader(args.path, args.dataset, args.batch_size)
    #model = models_.EfficientNetB0().to(device)
    #model = models.resnet101(weights='ResNet101_Weights.DEFAULT').to(device)
    model = models_.Wide_ResNet(depth=28, widen_factor=10, num_classes=10).to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    #optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()
    epoch = 300
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

    for epoch in range(start_epoch+1, start_epoch+epoch+1):
        train_tool.train(model, trainloader, optimizer, criterion, epoch, device)
        test_loss, test_accuracy = eval_tool.evaluate(model, testloader, criterion, device)
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

        print("\n[EPOCH: {}], \tModel: ResNet, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(epoch, test_loss, test_accuracy))
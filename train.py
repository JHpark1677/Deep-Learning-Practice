import argparse
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import torch.optim as optim

import models_
import dataloader

from tools import train_tool
from tools import eval_tool

def train():

    trainloader, testloader = dataloader.dataloader(args.path, args.dataset, args.batch_size)
    #model = models_.EfficientNetB0().to(device)
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT').to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        checkpoint = torch.load('../checkpoint/ckpt_2.pth')
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, betas(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = optim.SGD(model.parameters(), lr=0.1,
    #                  momentum=0.9, weight_decay=5e-4)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    #scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

    # to visualize
    train_losses = []
    val_losses = []
    batches = len(trainloader)
    best_acc, accuracy = 0, 0

    for epoch in range(start_epoch, start_epoch+1000):
        total_loss = 0.0
        progress = tqdm(enumerate(trainloader), total=batches)
        for i, data in progress:
            model.train()
            train_x, train_y = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(train_x)
            loss = criterion(outputs, train_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i==390: # how to set this ?
                total = 0
                train_losses.append(total_loss/300)
                model.eval()
                with torch.no_grad():
                    val_loss = 0
                    for j, val_data in enumerate(testloader):
                        val_x, val_y = val_data[0].to(device), val_data[1].to(device)
                        val_outputs = model(val_x)
                        v_loss = criterion(val_outputs, val_y)
                        val_loss += v_loss.item()
                        predicted_classes = torch.max(val_outputs, 1)[1]
                        total += val_y.size(0)
                        accuracy += (predicted_classes == val_y).sum().item()
                        val_losses.append(val_loss)

                    accuracy = (100 * accuracy / total)
                    print("  accuracy at {} iterations in {} epochs : {} ".format(i, epoch, accuracy))
                    print("  scheduler status | learning rate : {} ".format(optimizer.param_groups[0]['lr']))

                    if accuracy > best_acc :
                        print('Saving..')
                        state = {
                            'model' : model.state_dict(),
                            'optimizer' : optimizer.state_dict(), 
                            'acc' : accuracy,
                            'epoch' : epoch
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        torch.save(state, '../checkpoint/ckpt_2.pth')
                        best_acc = accuracy

        #scheduler.step()

if __name__ == "__main__": #import시에 함수만 실행될 수 있게하기 위해서. 직접 파일을 실행시켰을 때 if문이 참이 되어 문장이 수행된다.
   
    parser = argparse.ArgumentParser(description="Deep-Learning Practice")
    parser.add_argument(
        "--path", 
        default="../data",
        type=str,
        help='data path'
    )
    parser.add_argument(
        "--batch_size", 
        default=100, 
        type=int
    )
    parser.add_argument(
        "--dataset", 
        default="cifar10", 
        type=str
    )
    parser.add_argument(
        "--resume",
        '-r',
        action='store_true', 
        help='resume from checkpoint'
    )
    parser.add_argument(
        "--model",
        default='net',
        type=str,
        help='model name'
    )
    parser.add_argument(
        '--load_ckp',
        default='ckpt_cifar.pth',
        type=str, 
        help='checkpoint_name'
    )
    parser.add_argument(
        '--save_ckp',
        default='ckpt_cifar.pth',
        type=str, 
        help='checkpoint_name'
    )

    args = parser.parse_args()
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, testloader = dataloader.dataloader(args.path, args.dataset, args.batch_size)
    #model = models_.EfficientNetB0().to(device)
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT').to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epoch = 100
    test_accuracy = 0

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        path = '../checkpoint/' + os.path.join(args.load_ckp)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer = checkpoint['optimizer']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0
        best_acc = 0

    optimizer = optim.Adadelta(model.parameters(), lr=0.01)

    for epoch in range(start_epoch+1, start_epoch+epoch+1):
        train_tool.train(model, trainloader, optimizer, criterion, epoch, device)
        test_loss, test_accuracy = eval_tool.evaluate(model, testloader, criterion, device)

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
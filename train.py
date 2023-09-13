import argparse
import os
import torch
from tqdm import tqdm
import torch.nn as nn
from torchvision import models
import torch.optim as optim

#import models_
import dataloader

def train():

    trainloader, testloader = dataloader.dataloader(args.path, args.dataset, args.batch_size)
    model = models.resnet101(weights='ResNet101_Weights.DEFAULT').to(device)

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        checkpoint = torch.load('../checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, betas(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1, gamma=0.95)

    # to visualize
    epoch_num = 100
    train_losses = []
    val_losses = []
    batches = len(trainloader)
    best_acc, accuracy = 0, 0

    for epoch in range(epoch_num):
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

            if (i+1) % 300 == 0 : # how to set this ?
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
                            'acc' : accuracy,
                            'epoch' : epoch
                        }
                        if not os.path.isdir('checkpoint'):
                            os.mkdir('checkpoint')
                        torch.save(state, '../checkpoint/ckpt.pth')
                        best_acc = accuracy

        scheduler.step()

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
        default=128, 
        type=int
    )
    parser.add_argument(
        "--dataset", 
        default="cifar10", 
        type=str
    )
    parser.add_argument(
        "--resume",
        default=False,
        type=bool,
        help='resume with checkpoint'
    )
    parser.add_argument(
        "--model",
        default='net',
        type=str,
        help='model name'
    )
    args = parser.parse_args()
    
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train()
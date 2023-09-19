import argparse
import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import dataloader
from models_ import vit_model
from tools import train_tool
from tools import eval_tool


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
        '-r',
        action='store_true', 
        help='resume from checkpoint'
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
    trainloader, testloader = dataloader.dataloader(args.path, args.dataset, args.batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    patch_size = (4,4)
    dim = 128
    depth = 8
    num_heads = 8
    mlp_dim = 256
    dropout = 0.
    learning_rate = 0.001
    epoch = 10

    model = vit_model.ViT(image_shape = (3,32,32), patch_size = patch_size, num_classes = 10, dim = dim, num_heads = num_heads, depth = depth, mlp_dim = mlp_dim, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

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
        best_acc = 0
        start_epoch = 0
    
    best_acc = 0
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

        print("\n[EPOCH: {}], \tModel: ViT, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(epoch, test_loss, test_accuracy))
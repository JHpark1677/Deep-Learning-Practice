import os
import torch
import torch.nn as nn
import visdom

from tqdm.auto import tqdm
from dataloader import cifar_dataset
from models_ import vit_cnn_model
from tools import train_tool
from tools import eval_tool


if __name__ == "__main__":

    from config import get_args_parser
    import configargparse

    parser = configargparse.ArgumentParser('VIT', parents=[get_args_parser()])
    args = parser.parse_args()

    torch.cuda.is_available()
    trainloader, testloader = cifar_dataset.dataloader(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.visdom_true:
        vis = visdom.Visdom()
    else:
        vis = None
    
    patch_size = (4,4)
    dim = 128
    depth = 8
    num_heads = 8
    mlp_dim = 256
    dropout = 0.
    learning_rate = 0.001
    epoch = 50

    model = vit_cnn_model.ViT(image_shape = (3,32,32), patch_size = patch_size, num_classes = 10, dim = dim, num_heads = num_heads, depth = depth, mlp_dim = mlp_dim, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('../checkpoint'), 'Error : no checkpoint directory found'
        path = '../checkpoint/' + os.path.join(args.load_ckp)
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        best_acc = 0
        start_epoch = 0

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    best_acc = 0
    for epoch in range(start_epoch+1, start_epoch+epoch+1):
        train_tool.train(model, trainloader, optimizer, criterion, epoch, device, vis, args)
        test_loss, test_accuracy = eval_tool.evaluate(model, testloader, criterion, device, vis, args)

        if test_accuracy > best_acc :
            print('Saving..')
            state = {
                'model' : model.state_dict(),
                'acc' : test_accuracy,
                'epoch' : epoch
            }
            if not os.path.isdir('../checkpoint'):
                os.mkdir('../checkpoint')
            path = '../checkpoint/' + os.path.join(args.save_ckp)
            torch.save(state, path)
            best_acc = test_accuracy

        print("\n[EPOCH: {}], \tModel: ViT, \tTest Loss: {:.4f}, \tTest Accuracy: {:.2f} % \n".format(epoch, test_loss, test_accuracy))
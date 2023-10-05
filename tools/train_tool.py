from tqdm.auto import tqdm
import torch
from torch.autograd import Variable
from tools import mix_up

def train(model, train_loader, optimizer, criterion, epoch, DEVICE, vis, args):
    """
    Trains the model with training data.

    Do NOT modify this function.
    """
    batches = len(train_loader)
    model.train()
    tqdm_bar = tqdm(train_loader, total=batches)

    if args.mixup is not True:
        for batch_idx, (image, label) in enumerate(tqdm_bar):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))

    elif args.mixup is True:
        for batch_idx, (image, label) in enumerate(tqdm_bar):
            image, label_a, label_b, lam = mix_up.mixup_data(image.to(DEVICE), label.to(DEVICE), DEVICE, args.alpha)
            image, label_a, label_b = map(Variable, (image, label_a, label_b))
            optimizer.zero_grad()
            output = model(image)
            loss = mix_up.mixup_criterion(criterion, output, label_a, label_b, lam)
            loss.backward()
            optimizer.step()
            tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))


        # visdom report
        
        if (batch_idx % args.train_vis_step == 0 or batch_idx == len(train_loader) - 1) and args.rank == 0:
            if vis is not None:
            # loss plot
                vis.line(X=torch.ones((1, 1)).cpu() * batch_idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss]).unsqueeze(0).cpu(),
                         win='train_loss_' + args.save_ckp,
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='train_loss_{}'.format(args.save_ckp),
                                   legend=['Total Loss']))
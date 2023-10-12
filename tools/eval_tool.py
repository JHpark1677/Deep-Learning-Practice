import torch
from tqdm.auto import tqdm

def evaluate(model, test_loader, criterion, epoch, DEVICE, vis, args):
    """
    Evaluates the trained model with test data.

    Do NOT modify this function.
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for image, label in tqdm(test_loader):
            image = image.to(DEVICE)
            label = label.to(DEVICE)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    if vis is not None:
    # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                    Y=torch.Tensor([test_loss]).unsqueeze(0).cpu(),
                    win='test_loss_' + args.save_ckp,
                    update='append',
                    opts=dict(xlabel='step',
                            ylabel='Loss',
                            title='test_loss_{}'.format(args.save_ckp),
                            legend=['Total Loss']))

    if vis is not None:
    # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * epoch,  # step
                    Y=torch.Tensor([test_accuracy]).unsqueeze(0).cpu(),
                    win='test_accuracy' + args.save_ckp,
                    update='append',
                    opts=dict(xlabel='step',
                            ylabel='Accuracy',
                            title='test_accuracy_{}'.format(args.save_ckp),
                            legend=['Total accuracy']))

    return test_loss, test_accuracy
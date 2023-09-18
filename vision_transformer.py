import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm.auto import tqdm

def train(model, train_loader, optimizer, criterion, DEVICE):
    """
    Trains the model with training data.

    Do NOT modify this function.
    """
    model.train()
    tqdm_bar = tqdm(train_loader)
    for batch_idx, (image, label) in enumerate(tqdm_bar):
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        tqdm_bar.set_description("Epoch {} - train loss: {:.6f}".format(epoch, loss.item()))


def evaluate(model, test_loader, criterion, DEVICE):
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
    return test_loss, test_accuracy



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
        "--model",
        default='net',
        type=str,
        help='model name'
    )
    args = parser.parse_args()
    torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train()
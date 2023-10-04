import argparse
import datetime
import os
import sys
sys.path.append('..')

import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
import util
from nets import adabin_resnet_18

# Set seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# CIFAR-10 class names
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a binary neural network using BoostA strategy with sequential training.")
    parser.add_argument('--cpu', action='store_true', help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/', help='dataset path')
    parser.add_argument('--arch', action='store', default='adabin_resnet_18', help='the architecture for the network: adabin_resnet_18')
    parser.add_argument('--lr', action='store', default=0.01, type=float, help='the initial learning rate')
    parser.add_argument('--epochs', action='store', default='0', help='first train epochs', type=int)
    parser.add_argument('--retrain_epochs', action='store', default='100', help='re-train epochs', type=int)
    parser.add_argument('--save_name', action='store', default='first_model', help='save the first trained model', type=str)
    parser.add_argument('--load_name', action='store', default='first_model', help='load pretrained model', type=str)
    parser.add_argument('--root_dir', action='store', default='ensemble_models/', help='root dir for different experiments', type=str)
    parser.add_argument('--pretrained', action='store', default=None, help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

    args = parser.parse_args()
    print('==> Options:', args)
    return args


def main():
    args = parse_arguments()

    # Ensure dataset path is valid
    if not os.path.isfile(os.path.join(args.data, 'train_data.npy')):
        raise ValueError('Please provide a valid data path with --data <DATA_PATH>')

    # Load training and testing datasets
    trainset = data.CIFARDataset(root=args.data, train=True)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testset = data.CIFARDataset(root=args.data, train=False)
    testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

    # Model Setup and Initialization
    model = adabin_resnet_18.binary_resnet18()

    # Use CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # Optimization Setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        train(model, optimizer, criterion, device, trainloader, epoch)
        acc = test(model, device, testloader, criterion, args.save_name, best_acc)
        if acc > best_acc:
            best_acc = acc

    if args.evaluate:
        eval_test(model, device, testloader)

def train(model, optimizer, criterion, device, trainloader, epoch, sample_weights=torch.Tensor(np.ones((50000, 1)) / 50000.0)):
    model.train()  # Set model to training mode
    total_loss = 0
    correct = 0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch}")  # Define the progress bar
    for batch_idx, (data, target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Zero the gradients
        output = model(data)  # Forward pass
        loss = criterion(output, target)  # Loss calculation
        loss.backward()  # Backward pass
        optimizer.step()  # Optimization step
        total_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(target).sum().item()
        progress_bar.set_description(f"Epoch {epoch} Loss: {loss.item():.6f} LR: {optimizer.param_groups[0]['lr']:.6f}")
    accuracy = 100. * correct / len(trainloader.dataset)
    print(f'Training Accuracy: {accuracy:.2f}%')
    return total_loss / len(trainloader)  # Return average loss

def test(model, device, testloader, criterion, save_name, best_acc):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            total_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%')
    if accuracy > best_acc:
        print(f'Saving model as {save_name}')
        torch.save(model.state_dict(), save_name)
    return accuracy


def eval_test(model, device, testloader):
    model.eval()  # Set model to evaluation mode
    correct = 0
    with torch.no_grad():  # Disable gradient computation
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # Forward pass
            _, predicted = output.max(1)
            correct += predicted.eq(target).sum().item()
    accuracy = 100. * correct / len(testloader.dataset)
    print(f'Evaluation Accuracy: {accuracy:.2f}%')
    return accuracy


def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
    pass


if __name__ == '__main__':
    main()

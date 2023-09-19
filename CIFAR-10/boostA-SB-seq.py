# ----------------------------------
# Imports and Setup
# ----------------------------------

import torch
import cv2
import time
import datetime
import os
import argparse
import sys
import numpy as np
from torchvision import datasets, transforms
import data
import util
from models import nin
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Set seed for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# CIFAR-10 class names
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ----------------------------------
# Argument Parsing
# ----------------------------------

parser = argparse.ArgumentParser(description="Train a binary neural network using BoostA strategy with sequential training.")
parser.add_argument('--cpu', action='store_true', help='set if only CPU is available')
parser.add_argument('--data', action='store', default='./data/', help='dataset path')
parser.add_argument('--arch', action='store', default='nin', help='the architecture for the network: nin')
parser.add_argument('--lr', action='store', default='0.01', help='the intial learning rate')
parser.add_argument('--epochs', action='store', default='0', help='fisrt train epochs',type=int)
parser.add_argument('--retrain_epochs', action='store', default='100', help='re-train epochs',type=int)
parser.add_argument('--save_name', action='store', default='first_model', help='save the first trained model',type=str)
parser.add_argument('--load_name', action='store', default='first_model', help='load pretrained model',type=str)
parser.add_argument('--root_dir', action='store', default='models_boostA_SB_seq/', help='root dir for different experiments',type=str)
parser.add_argument('--pretrained', action='store', default=None, help='the path to the pretrained model')
parser.add_argument('--evaluate', action='store_true', help='evaluate the model')

args = parser.parse_args()
print('==> Options:', args)

# ----------------------------------
# Data Preparation
# ----------------------------------

# Ensure dataset path is valid
if not os.path.isfile(args.data + '/train_data'):
    raise ValueError('Please provide a valid data path with --data <DATA_PATH>')

# Load training and testing datasets
trainset = data.dataset(root=args.data, train=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4)
testset = data.dataset(root=args.data, train=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False, num_workers=4)

# ----------------------------------
# Model Setup and Initialization
# ----------------------------------

print(f'==> Building model {args.arch}...')
if args.arch == 'nin':
    model = nin.Net()
else:
    raise ValueError(f'{args.arch} is currently not supported')

# Model initialization or loading pre-trained model
if not args.pretrained:
    print('==> Initializing model parameters ...')
    # ... [model initialization code]
else:
    print(f'==> Load pretrained model from {args.pretrained}...')
    pretrained_model = torch.load(args.pretrained)
    model.load_state_dict(pretrained_model['state_dict'])

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
if torch.cuda.is_available():
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))


# ----------------------------------
# Optimization Setup
# ----------------------------------

# Set up optimizer parameters
param_dict = dict(model.named_parameters())
params = []
base_lr = args.lr

for key, value in param_dict.items():
    if 'bias' in key:
        params += [{'params': [value], 'lr': 2 * base_lr, 'weight_decay': 0}]
    else:
        params += [{'params': [value], 'lr': base_lr, 'weight_decay': 0.00001}]
optimizer = optim.Adam(params, lr=base_lr, weight_decay=0.00001)


# Set up loss functions
criterion = nn.CrossEntropyLoss()
criterion_separated = nn.CrossEntropyLoss(reduction='none')

# Set up binarization operator
bin_op = util.BinOp(model, args.arch)

# ----------------------------------
# Functions: Training, Testing, etc.
# ----------------------------------

def adjust_learning_rate(optimizer, epoch):
    """Adjusts the learning rate during training."""
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

def reset_learning_rate(optimizer, lr=0.01):
    """Resets the learning rate to a specified value."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_state(model, best_acc, save_name, root_dir=args.root_dir):
    """Saves the model state."""
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    torch.save(state, os.path.join(root_dir, save_name + '.pth.tar'))

def train(epoch, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
    """Trains the model for one epoch."""
    adjust_learning_rate(optimizer, epoch)
    model.train()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights[:,0].double(), 50000)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=4, sampler=sampler)
    
    for batch_idx, (data, target) in enumerate(trainloader):
        bin_op.binarization()
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data) 
        loss = criterion(output, target)
        loss.backward()
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(trainloader.dataset)} ({100. * batch_idx / len(trainloader):.0f}%)]\tLoss: {loss.item():.6f}\tLR: {optimizer.param_groups[0]["lr"]:.6f}')
    return trainloader 

def test(save_name, best_acc, sample_weights=torch.Tensor(np.ones((50000,1))/50000.0)):
    """Tests the model."""
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights[:,0].double(), 50000)
    testloader_in = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=False, num_workers=4, sampler=sampler)
    
    with torch.no_grad():
        for data, target in testloader_in:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    bin_op.restore()
    acc = 100. * correct / len(testloader_in.dataset)
    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc, save_name)
    
    test_loss = test_loss / len(testloader_in.dataset) * 1000
    print(f'\nTrain set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader_in.dataset)} ({acc:.2f}%)')
    print(f'Best Train Accuracy: {best_acc:.2f}%\n')
    return best_acc

def eval_test():
    """Evaluates the model on the test set."""
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    
    with torch.no_grad():
        for data, target in testloader:
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    bin_op.restore()
    acc = 100. * correct / len(testloader.dataset)
    test_loss /= len(testloader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(testloader.dataset)} ({acc:.2f}%)')

def sample_models(boosting_iters, sample_weights, retrain_epoch=100):
    """Train a model with given sample weights and return updated weights."""
    print(f'{str(datetime.datetime.utcnow())} Start boosting iteration: {boosting_iters}')
    print('===> Start retraining...')
    
    best_acc = 0
    reset_learning_rate(optimizer, lr=0.01)
    for epoch in range(1, retrain_epoch + 1):
        train(epoch, sample_weights)
        if epoch % 5 == 0:
            best_acc = test(str(boosting_iters), best_acc, sample_weights)
            eval_test()

    pretrained_model = torch.load(os.path.join(args.root_dir, f"{boosting_iters}.pth.tar"))
    best_acc = pretrained_model['best_acc']
    model.load_state_dict(pretrained_model['state_dict'])
    model.eval()
    bin_op.binarization()

    pred_output = torch.zeros((50000, 1)) # torch tensor on CPU
    label_in_tensor = torch.zeros((50000, )) # torch tensor on CPU
    for batch_idx, (data, target) in enumerate(trainloader):
        batch_size = target.size(0)
        batch_sample_weights = sample_weights[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        batch_softmax_output = get_error_output(data, target, batch_sample_weights)
        pred_output[batch_idx * batch_size: (batch_idx + 1) * batch_size, :] = batch_softmax_output.argmax(dim=1, keepdim=True).cpu()
        label_in_tensor[batch_idx * batch_size: (batch_idx + 1) * batch_size] = target

    bin_op.restore()
    best_sample_weights, best_alpha_m = update_weights(pred_output, label_in_tensor, sample_weights)

    return best_sample_weights, best_alpha_m


def use_sampled_model(sampled_model, data, target, batch_sample_weights):
    """Use a pretrained model to get the output for given data."""
    pretrained_model = torch.load(sampled_model)
    model.load_state_dict(pretrained_model['state_dict'])
    model.eval()
    bin_op.binarization()
    
    return get_error_output(data, target, batch_sample_weights)


def combine_softmax_output(pred_test_i, pred_test, alpha_m_mat, i):
    """Combine softmax outputs using the alpha values from boosting."""
    pred_test_delta = alpha_m_mat[0][i] * pred_test_i
    pred_test = torch.add(pred_test, pred_test_delta.cpu())
    return pred_test

def get_error_output(data, target, batch_sample_weights):
    """Compute the error output for the given data and target."""
    data, target = data.to(device), target.to(device)
    output = model(data)
    
    # Compute the loss using the sample weights
    loss = (criterion_separated(output, target) * batch_sample_weights.to(device)).mean()
    
    return output

def update_weights(softmax_output, target, sample_weights):
    """Update the sample weights based on the error."""
    pred_numpy = softmax_output
    target_numpy = target
    pred_numpy = torch.squeeze(pred_numpy)
    miss = torch.Tensor([int(x) for x in (pred_numpy != target_numpy)])
    miss2 = torch.Tensor([x if x==1 else -1 for x in miss])
    miss = miss.unsqueeze(1)

    err_m = torch.mm(torch.t(sample_weights), miss) / torch.sum(sample_weights)
    alpha_m = 0.5 * torch.log((1 - err_m) / float(err_m))

    prior_exp = torch.t(alpha_m * miss2)
    sample_weights_new = sample_weights * torch.exp(prior_exp)

    return sample_weights_new, alpha_m


def most_common_element(pred_mat, alpha_m_mat, num_boost):
    """Determine the most common prediction using boosting alpha values."""
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(1000):
        best_value = -1000
        best_pred = -1
        for j in range(10):
            mask = [int(x) for x in (pred_mat[i, :] == j * np.ones((num_boost,), dtype=int))]
            if np.sum(mask * alpha_m_mat[0, :num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0, :num_boost])
                best_pred = j
        pred_most.append(best_pred)

    return np.array(pred_most)

def most_common_element_train(pred_mat, alpha_m_mat, num_boost):
    pred_most = []
    pred_mat = pred_mat.astype(int)
    for i in range(128):
        best_value = -1000
        best_pred = -1
        for j in range(10):
            mask = [int(x) for x in (pred_mat[i, :] == j * np.ones((num_boost,), dtype=int))]
            if np.sum(mask * alpha_m_mat[0, :num_boost]) > best_value:
                best_value = np.sum(mask * alpha_m_mat[0, :num_boost])
                best_pred = j
        pred_most.append(best_pred)

    return np.array(pred_most)


# ----------------------------------
# Main Execution Logic
# ----------------------------------

if __name__ == '__main__':
    # Train the model if specified number of epochs is not zero
    if args.epochs != 0:
        for epoch in range(1, int(args.epochs) + 1):
            train(epoch)
            best_acc = test(args.save_name, best_acc)
    else:
        # Load pretrained model if no training epochs are specified
        print('==> Loading pretrained model...')
        pretrained_model = torch.load(os.path.join(args.root_dir, args.load_name + '.pth.tar'))
        model.load_state_dict(pretrained_model['state_dict'])

    boosting_iters = 32
    # Initialize sample weights equally for all samples
    sample_weights_new = torch.Tensor(np.ones((50000, 1)) / 50000.0)
    index_weak_cls = 0
    alpha_m_mat = torch.Tensor()

    # Boosting: iteratively train weak models and adjust sample weights
    for i in range(boosting_iters):
        print(f"Boosting iteration: {i}")
        sample_weights_new, alpha_m = sample_models(boosting_iters=i, sample_weights=sample_weights_new, retrain_epoch=args.retrain_epochs)
        print(f'{str(datetime.datetime.utcnow())} {i}-th Sample done!')
        
        index_weak_cls += 1
        alpha_m_mat = torch.cat((alpha_m_mat, alpha_m), 1)

    print("Boosting finished!")

    pred_store = torch.Tensor()

    # Use the boosted models
    for num_boost in range(1, 32):
        final_loss_total = []
        final_loss_total_2 = []
        for data, target in testloader:
            pred_store = torch.Tensor()
            pred_test = torch.Tensor(np.zeros((1000, 10)))
            
            for i in range(num_boost):
                pred_test_i = use_sampled_model(os.path.join(args.root_dir, f"{i}.pth.tar"), data, target, torch.Tensor(np.ones((1000, 1)) / 1.0))
                pred = pred_test_i.argmax(dim=1, keepdim=True)
                pred_store = torch.cat((pred_store, pred.data.cpu().float()), 1)
                pred_test = combine_softmax_output(pred_test_i.data.cpu(), pred_test, alpha_m_mat, i)

            pred_most = most_common_element(pred_store.numpy(), alpha_m_mat.numpy(), num_boost)
            pred_most = torch.Tensor(pred_most)
            pred_test = pred_test.argmax(dim=1, keepdim=True).squeeze(1)

            loss_final = [int(x) for x in (pred_most.numpy() != target.numpy())]
            final_loss = float(sum(loss_final)) / float(len(loss_final))
            final_loss_total.append(final_loss)

            loss_final_2 = [int(x) for x in (pred_test.numpy() != target.numpy())]
            final_loss_2 = float(sum(loss_final_2)) / float(len(loss_final_2))
            final_loss_total_2.append(final_loss_2)

        print(f'-------------------- Boosting Iteration: {num_boost} --------------------')
        final_loss_print = np.mean(final_loss_total)
        print(f'\nTest accuracy from selected model: {1-final_loss_print:.4f}')
        
        final_loss_print_2 = np.mean(final_loss_total_2)
        print(f'\nTest accuracy from selected model 2: {1-final_loss_print_2:.4f}')

    os._exit(0)


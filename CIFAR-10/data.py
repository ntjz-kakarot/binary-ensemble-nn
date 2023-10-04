import os
import torch
import numpy as np
import torchvision.transforms as transforms

class CIFARDataset(torch.utils.data.Dataset):
    """
    A dataset class for loading CIFAR data.
    """

    def __init__(self, root=None, train=True, example_weight=None):
        """
        Initialize the CIFAR dataset.

        Parameters:
        - root (str): Path to the dataset directory.
        - train (bool): If True, initializes training data, else test data.
        - example_weight (list): Weights for each example, used for weighted sampling.
        """
        self.root = root
        self.train = train
        self.transform = transforms.ToTensor()  # Convert images to PyTorch tensors.

        file_extension = ".npy"
        data_path = os.path.join(root, 'train_data' + file_extension) if train else os.path.join(root, 'test_data' + file_extension)
        labels_path = os.path.join(root, 'train_labels' + file_extension) if train else os.path.join(root, 'test_labels' + file_extension)

        self.data = np.load(open(data_path, 'rb'))
        self.data = torch.from_numpy(self.data.astype('float32'))
        self.labels = np.load(open(labels_path, 'rb')).astype('int')
        self.example_weight = example_weight

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve a single sample from the dataset.

        Parameters:
        - index (int): Index of the sample to retrieve.

        Returns:
        - img (tensor): The image sample.
        - target (int): The label of the sample.
        - weight (float): The weight of the sample (only if in training mode).
        """
        if self.train:
            if self.example_weight is not None:
                img, target, weight = self.data[index], self.labels[index], self.example_weight[index]
                return img, target, weight
            else:
                img, target = self.data[index], self.labels[index]
                return img, target
        else:
            img, target = self.data[index], self.labels[index]
            return img, target

def build_test_loader(root='./data/', batch_size=1000, shuffle=False, num_workers=4):
    """
    Construct a DataLoader for the test dataset.

    Parameters:
    - root (str): Path to the dataset directory.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): If True, shuffles the dataset.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - DataLoader object for the test dataset.
    """
    test_set = CIFARDataset(root=root, train=False)
    return torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def build_train_loader(mode, example_weight, root='./data/', batch_size=128, shuffle=False, num_workers=4):
    """
    Construct a DataLoader for the training dataset with different sampling modes.

    Parameters:
    - mode (str): Sampling mode ('normal', 'sample_batch', or 'sample_dataset').
    - example_weight (list): Weights for each example, used for weighted sampling.
    - root (str): Path to the dataset directory.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): If True, shuffles the dataset.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - DataLoader object for the training dataset based on the specified mode.
    """
    train_set = CIFARDataset(root=root, train=True, example_weight=example_weight)
    
    if mode == 'normal':
        # Regular data loading without any weights.
        return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    elif mode == 'sample_batch':
        # Sample batches based on the provided example weights.
        sampler = torch.utils.data.sampler.WeightedRandomSampler(example_weight.double(), len(train_set))
        return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    
    elif mode == 'sample_dataset':
        # Sample from a subset of indices without replacement.
        indices = example_weight
        sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        return torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, sampler=sampler)
    
    else:
        raise ValueError(f"Undefined Mode: {mode}")

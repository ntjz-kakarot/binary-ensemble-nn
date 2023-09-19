import numpy as np
import os
import pickle

def unpickle(file):
    """Load byte data from file"""
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin-1')
    return data

def load_cifar10_data(data_dir):
    """Load CIFAR-10 data from `data_dir`"""
    train_data = None
    train_labels = []
    
    # Load all training data (5 batches)
    for i in range(1, 6):
        data_dic = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = data_dic['data']
        else:
            train_data = np.vstack((train_data, data_dic['data']))
        train_labels += data_dic['labels']

    test_data_dic = unpickle(data_dir + "/test_batch")
    test_data = test_data_dic['data']
    test_labels = test_data_dic['labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    test_data = test_data.reshape((len(test_data), 3, 32, 32))

    return train_data, train_labels, test_data, test_labels

def process_and_save_data():
    # Adjust this path to the location of your CIFAR-10-PY-Dataset directory
    data_dir = '../../CIFAR-10-PY-Dataset'
    
    train_data, train_labels, test_data, test_labels = load_cifar10_data(data_dir)
    
    # Create a 'data' directory inside 'CIFAR-10' if it doesn't exist
    save_dir = './data'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save as numpy arrays
    np.save(os.path.join(save_dir, 'train_data.npy'), train_data)
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_labels)
    np.save(os.path.join(save_dir, 'test_data.npy'), test_data)
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_labels)

    print("Data processed and saved to './data/' directory.")

if __name__ == '__main__':
    process_and_save_data()

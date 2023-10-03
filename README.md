An attempt to recreate updated implementation from [BENN-PyTorch](https://github.com/XinDongol/BENN-PyTorch/tree/master)

# Instructions to run the code

1. Create and prepare the environment:

```python
conda create --name benn_env python=3.11
conda activate benn_env
```

2. Download [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset, and 

Below is a snapshot of the directory strucutre required to run the code. Make sure to have the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset downloaded as shown in *CIFAR-10-PY-Dataset*
```
├── binary-ensemble-nn
│   ├── CIFAR-10
│   │   ├── data
│   │   │   ├── test_data.npy
│   │   │   ├── test_labels.npy
│   │   │   ├── train_data.npy
│   │   │   └── train_labels.npy
│   │   ├── data.py
│   │   ├── main-boostA-SB-seq.py
│   │   ├── models
│   │   │   ├── nin.py
│   │   │   └── __pycache__
│   │   │       └── nin.cpython-38.pyc
│   │   ├── models_boostA_SB_seq
│   │   │   └── 0.pth.tar
│   │   ├── process_cifar10.py
│   │   └── __pycache__
│   │       └── data.cpython-38.pyc
│   ├── __pycache__
│   │   └── util.cpython-38.pyc
│   ├── README.md
│   └── util.py
└── CIFAR-10-PY-Dataset
    ├── batches.meta
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── readme.html
    └── test_batch
```

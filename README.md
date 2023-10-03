An attempt to recreate updated implementation from [BENN-PyTorch](https://github.com/XinDongol/BENN-PyTorch/tree/master)

### Instructions to run the code

Create and prepare the environment:

```python
conda create --name benn_env python=3.11
conda activate benn_env
```
Install libraries

```python
conda install future numpy pillow pyyaml six
```

Install Pytorch-CUDA (I have tested with both 11.8(Stable) and 12.1(Nightly))

```python
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Download and process the dataset

The below script will download and process the dataset

```python
python process_cifar10.py
```

### Train the model

```python
python main-boostA-SB-seq.py --{OPTIONS}
```

Default options 
cpu=False, data='./data/', arch='nin', lr=0.01, epochs=0, retrain_epochs=100, save_name='first_model', load_name='first_model', root_dir='models_boostA_SB_seq/', pretrained=None, evaluate=False
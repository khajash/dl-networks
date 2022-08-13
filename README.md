# Deep Learning Network Implementations
This repo will have various deep learning network implementations and data loaders. The networks will cover a handful of models for image classification, image segmentation, and generative networks. I use PyTorch for networks/training and am logging metrics using Weights and Biases.  

## Datasets
- **Classification:** 
    - [Imagenette](https://github.com/fastai/imagenette) - For testing networks and training at home on my old machine, this is great to play around with. It's a small subset of Imagenet (only 10 classes) and has full resolution images. 

## Networks
-  **Classification:**
    - AlexNet [(Krizhevsky et al., 2012)](https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
    - VGG [(Simonyan & Zisserman, 2015)](https://arxiv.org/abs/1409.1556)
    - ResNet [(He et al., 2015)](https://arxiv.org/abs/1512.03385)
- **Segmentation:**
    - UNet [(Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)
## Notebooks
- [Custom-Pytorch-Dataset_Imagenette.ipynb](https://github.com/khajash/dl-networks/blob/main/notebooks/Custom-Pytorch-Dataset_Imagenette.ipynb) - Notebook walks through implementing a custom PyTorch dataset with the Imagenette dataset.

## Setup
- Recommended to use a virtual environment, such as `venv`, `virtualenv` or `conda`

```
git clone https://github.com/khajash/dl-networks.git
cd dl-networks
python -m venv .env
source .env/bin/activate
pip install -e .
```

## Usage
- To use Imagenette dataset, dowload from [repo here](https://github.com/fastai/imagenette).
- Create a config file for your network of choice in `dlnets/models/configs` 
-  The `model` argument in the command line is used as the wandb group and also selects the model class to initialize. Examples include: `ALEXNET`, `VGG11`, `RESNET_SMALL`, `RESNET18`. Use the config file to choose the network parameters, changing the value in the model name (e.g. RESNET18, VGG11) does not currently change the network parameters.

```
cd dlnets/train
python train_classification.py --datadir path/to/imagenette2 --model VGG11 --yaml ../models/configs/config-vgg-small.yaml
```
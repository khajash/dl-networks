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
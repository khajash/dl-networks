import os
import numpy as np
# import pandas as pd
# from skimage import io, transform
# import matplotlib.pyplot as plt
# from PIL import Image
# import random
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import wandb

# TODO: create a model and dataset directory
from data.imagenette import ImagenetteDataset, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from dl.vision.alexnet import AlexNet

torch.manual_seed(17)
np.random.seed(17)

def get_imagenette_datasets(datadir, noisy_perc=0):
    data_transforms = {
        "train" : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            lambda x : x / 255.,
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
            # TODO: color agumentation
        ]),
        "test" : transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            lambda x : x / 255.,
            transforms.Normalize(mean=IMAGENET_RGB_MEANS, std=IMAGENET_RGB_STDS),
        ])
    }

    # one hot encoding transformation
    target_transform = transforms.Lambda(
        lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))

    train_dataset = ImagenetteDataset(root_dir = datadir,
        csv_filename = "noisy_imagenette.csv",
        train = True,
        noisy_perc = noisy_perc,
        transform = data_transforms["train"],
        target_transform = target_transform)

    test_dataset = ImagenetteDataset(root_dir = datadir,
        csv_filename = "noisy_imagenette.csv",
        train = False,
        noisy_perc = noisy_perc,
        transform = data_transforms["test"],
        target_transform = target_transform)
    
    return train_dataset, test_dataset


def train_loop(dataloader, model, loss_fn, optimizer):

    d_size = len(dataloader.dataset)
    model.train()
    print(f"{len(dataloader)} training batches")
    for batch_idx, (data, label) in enumerate(dataloader):
        output = model(data)
        loss = loss_fn(output, label)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            train_acc = (output.argmax(1) == label).type(torch.float).sum().item()
            wandb.log({"train_accuracy":train_acc, "train_loss": loss})
            print(f"loss: {loss:>7f}  [{current:>5d}/{d_size:>5d}]")



def test_loop(dataloader, model, loss_fn):
    d_size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, test_acc = 0, 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            output = model(data)
            test_loss += loss_fn(output, label)
            test_acc += (output.argmax(1) == label).type(torch.float).sum().item()

    test_loss /= n_batches
    test_acc /= d_size
    print(f"Test Error: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", 
        default="../../datasets/imagenette2", 
        type=str,
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed. (int, default = 0)",
    )
    parser.add_argument(
        "--n_epochs",
        default=100,
        type=int,
        help="Number of epochs to run the training. (int, default = 100)",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="Batch size for mini-batch training. (int, default = 20)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate. (float, default = 1e-4)",
    )

    return parser.parse_args()

def main():
    
    args = setup_training_parser()

    config = vars(args)
    config.update(momentum=0.9, weight_decay=5e-4, dataset="Imagenette", network="AlexNet")
    
    # Maybe separate Project: Imagenette and Group: AlexNet-v0
    wandb.init(project="Imagenette-AlexNet-v1", config=config)
    # use wandb.config.update({}) to update hyperparameters and other configs you want to save
    
    # setup data
    train_dataset, test_dataset = get_imagenette_datasets(args.datadir, 0)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # setup model
    model = AlexNet(in_channels=3, num_classes=10)

    # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
    loss = nn.CrossEntropyLoss()
    
    # setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr, 
        momentum=config["momentum"], 
        weight_decay=config["weight_decay"])
    
    for i in tqdm(range(args.n_epochs)):
        print(f"Epoch {i}\n--------------------------------")
        train_loop(train_dataloader, model, loss, optimizer)
        test_loop(test_dataloader, model, loss)

import os
import numpy as np
from tqdm import tqdm
import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils

import wandb

# TODO: create a model and dataset directory
from dlnets.data.imagenette import ImagenetteDataset, IMAGENET_RGB_MEANS, IMAGENET_RGB_STDS
from dlnets.models.vision import AlexNet, VGG, ResNet, ResNetBase

def get_imagenette_datasets(datadir, noisy_perc=0, device="cpu"):
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
        target_transform = target_transform,
        device=device)

    test_dataset = ImagenetteDataset(root_dir = datadir,
        csv_filename = "noisy_imagenette.csv",
        train = False,
        noisy_perc = noisy_perc,
        transform = data_transforms["test"],
        target_transform = target_transform,
        device=device)
    
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

        if batch_idx % 25 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            # print("acc output shapes: ", output.shape, label.shape)
            train_acc = (output.argmax(1) == label.argmax(1)).type(torch.float).mean().item()
            wandb.log({"train_accuracy":train_acc, "train_loss": loss})
            print(f"loss: {loss:>7f}  acc: {train_acc:>7f}  [{current:>5d}/{d_size:>5d}]")


def test_loop(dataloader, model, loss_fn, scheduler):
    d_size = len(dataloader.dataset)
    n_batches = len(dataloader)
    test_loss, test_acc = 0, 0
    model.eval()
    print(f"{len(dataloader)} test batches")
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            output = model(data)
            test_loss += loss_fn(output, label)
            test_acc += (output.argmax(1) == label.argmax(1)).type(torch.float).sum().item()

    test_loss /= n_batches
    test_acc /= d_size
    print(f"Test Error: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    wandb.log({"test_accuracy":test_acc, "test_loss": test_loss})
    
    scheduler.step(test_loss)


def setup_training_parser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", 
        default="../../..s/datasets/imagenette2", 
        type=str,
    )
    parser.add_argument(
        "--model",
        default="VGG11", 
        type=str,
        help="Model Name",
    )
    parser.add_argument(
        "--yaml",
        default="../models/configs/config-vgg-small.yaml", 
        type=str,
        help="Save model when done.",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Random seed. (int, default = 0)",
    )
    parser.add_argument(
        "--n_epochs",
        default=75,
        type=int,
        help="Number of epochs to run the training. (int, default = 75)",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size for mini-batch training. (int, default = 64)",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum used in SGD optimizer. (float, default = 0.9)",
)
    parser.add_argument(
        "--decay",
        default=5e-4,
        type=float,
        help="Weight Decay used in SGD optimizer. (float, default = 5e-4)",
    )
    parser.add_argument(
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate. (float, default = 1e-4)",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save model when done.",
    )

    return parser.parse_args()

def load_model(model_name, net_config, in_channels=3, num_classes=10):
    model_name = model_name.upper()
    if model_name == "ALEXNET":
        return AlexNet(in_channels, num_classes)
    elif model_name[:3] == "VGG":
        return VGG(in_channels, num_classes, **net_config)
    elif model_name == "RESNET_SMALL":
        return ResNetBase(**net_config)
    elif model_name[:3] == "RES":
        return ResNet(**net_config)

def main():
    
    args = setup_training_parser()
    default_config = vars(args)

    with open(args.yaml, "r") as f:
        net_config = yaml.safe_load(f)
    print(net_config)

    # add notes, tags, 
    default_config.update(
        dataset="Imagenette", 
        network=args.model, 
        **net_config)
    
    # Setup wandb configuration
    wandb.init(project="Imagenette", group=f"{args.model}-v0", config=default_config)
    config = wandb.config
    print("\nConfig:\n", config)
    # use wandb.config.update({}) to update hyperparameters and other configs you want to save

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    
    # setup data
    train_dataset, test_dataset = get_imagenette_datasets(config.datadir, 0, device)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

    # setup model
    model = load_model(config.model, net_config)

    print(model)
    print()

    model.to(device)

    # track gradients
    wandb.watch(model, log_freq=50)

    # nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss
    loss = nn.CrossEntropyLoss()
    
    # setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=config.lr, 
        momentum=config.momentum, 
        weight_decay=config.decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1)
    wandb.config.update({"scheduler": scheduler})
    
    for i in tqdm(range(config.n_epochs)):
        print(f"Epoch {i}\n--------------------------------")
        train_loop(train_dataloader, model, loss, optimizer)
        test_loop(test_dataloader, model, loss, scheduler)
        wandb.log({"lr": optimizer.param_groups[0]['lr']})
        if i % 5 == 0 and config.save_model:
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == "__main__":
    main()
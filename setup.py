#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
      name="dlnets",
      version="0.0.1",
      install_requires=[
            "torch",
            "torchvision",
            "numpy",
            "tqdm",
            "wandb"
      ],
      description="This repo contains reimplementations of deep learning models.",
      author="Kate Hajash",
      url="https://github.com/khajash/dl-networks",
      author_email="kshajash@gmail.com",
      packages=find_packages(exclude=['notebooks']),
)

# MLVISION

A pytorch train library

## Introduction

Most classification tasks do not require to rewrite a train loop in pytorch, using this library allows for fast iteration using any deep learning architecture.

The library can be use for cpu, gpu, and multi gpu training, and implements an efficient train loop as well as logging in order to plot train history.

## Installation

You can install it using poetry by running:

```
poetry install
```

## Usage

### Train a Model

To train a model you will need to edit a config file with at least the path of the training and validation data.

For more details on the config you can take a look at the next section.

You can now run the train command using:
```sh
# One GPU
poetry run python train.py --config /path/to/config --devices cuda:0
```

The code support multi gpu training using DDP:
```sh
poetry run python train.py --config /path/to/config --devices cuda:0 cuda:1 cuda:2 ...
```

or cpu training:
```sh
poetry run python train.py --config /path/to/config --devices cpu
```

### Config

The config allows you to change multiple hyper parameter of the training:
* data
  * path to dataset
  * data augmentation
  * workers for data loaders
* optimization
  * number of epochs
  * learning rate
  * float16 / float32
* model
  * Architecture (defined in `mlvision.models.__init__.py`)
  * args: given as kwargs to the model class

### Add a new model

You can easily define a new model and use it, to do so create a new file in `mlvision/models` and add the class in the `mlvision.models.__init__.py` file.

You should now be able to use it in a config file.

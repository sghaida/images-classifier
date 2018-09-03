import json
import os
from collections import OrderedDict

import torch
from torch import nn, optim
from torchvision import models

"""
this is a utility to load and save models
"""


def load_model(categories_json, hidden_units: list=None, arch='vgg16'):
    """
    :param categories_json: json file which holds categories
    :param arch: model architecture. default  arch='vgg16'
    :param hidden_units: number of hidden units
    :return: the model with the new classifier configs
    """

    # supported archs
    archs = ['vgg11', 'vgg13', 'vgg16', 'vgg19', 'densenet121', 'densenet169', 'densenet161', 'densenet201']

    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    # load categories
    with open(categories_json, 'r') as f:
        cat_to_name = json.load(f)

    if arch not in archs:
        raise ValueError('Unexpected network architecture', arch)

    # initialize related variables
    output_size = len(cat_to_name)

    if hidden_units is None:
        hidden_sizes = [3136, 784]
    else:
        hidden_sizes = hidden_units

    od = OrderedDict()

    loaded_model = models.__dict__[arch](pretrained=True)

    # Input size from current classifier if VGG
    if arch.startswith("vgg"):
        input_size = loaded_model.classifier[0].in_features
    else:
        input_size = densenet_input[arch]

    hidden_sizes.insert(0, input_size)

    # Prevent back propagation on parameters
    for param in loaded_model.parameters():
        param.requires_grad = False

    for i in range(len(hidden_sizes) - 1):
        od[f'fc{i+1}'] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        od[f'relu{i+1}'] = nn.ReLU()
        od[f'dropout{i+1}'] = nn.Dropout(p=0.2)

    od['output'] = nn.Linear(hidden_sizes[len(hidden_sizes)-1], output_size)
    od['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)

    # Replace classifier
    loaded_model.classifier = classifier

    return loaded_model


def save_model(
        model_directory: str, trained_model: models, class_to_idx: dict, optimizer: optim, arch: str,
        epochs=4, model_name: str='checkpoint.pth'):
    """
    Saves model to directory
    :param model_directory: a path where the model should be saved
    :param trained_model: the model to be saved
    :param class_to_idx: Dict with items (class_name, class_index)
    :param optimizer: the optimizer that has been used in training
    :param arch:
    :param epochs: number of epochs that is used in the training. could be used later
    :param model_name: checkpoint name
    :return:
    """

    # check for save directory
    if not os.path.isdir(model_directory):
        print(f'Directory {model_directory} does not exist. Creating...')
        os.makedirs(model_directory)

    trained_model.class_to_idx = class_to_idx

    model_state = {
        'epoch': epochs,
        'state_dict': trained_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': trained_model.classifier,
        'class_to_idx': trained_model.class_to_idx,
        'arch': arch
    }

    save_location = f'{model_directory}/{model_name}'
    print(f"Saving checkpoint to {save_location}")

    torch.save(model_state, save_location)


if __name__ == '__main__':
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    model = load_model(categories_json=f"{ROOT_DIR}/../cat_to_name.json", arch="vgg16")

    print(model)

import os
import torch.optim as optim
import argparse

from utils.loader import load_model as model_loader, save_model
from training.train import Training


# Define command line arguments

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset ')
parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
parser.add_argument('--arch', type=str, default='vgg16', help='Model architecture')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_units', nargs='+', type=int, default=[3136, 784], help='Number of hidden units')
parser.add_argument('--save_dir', type=str, default='.', help='Save trained model checkpoint to file in save_dir path')

args, _ = parser.parse_known_args()


# train and save model
def train_model():

    root_dir = os.path.dirname(os.path.abspath(__file__))

    print(f'loading the model with the following parameters')
    print(f'arch={args.arch}')
    print(f'hidden_units={args.hidden_units}')
    print(f'-----------------------------')

    model = model_loader(
        categories_json=f"{root_dir}/cat_to_name.json",
        arch=args.arch,
        hidden_units=args.hidden_units
    )

    model_trainer = Training(
        expected_means=[0.485, 0.456, 0.406],
        expected_std=[0.229, 0.224, 0.225],
        max_image_size=224,
        batch_size=32,
        data_directory=f'{args.data_dir}'
    )

    if args.gpu:
        use_gpu = True
    else:
        use_gpu = False

    print(f'training {args.arch} model with the following parameters')
    print(f'learning_rate={args.learning_rate}')
    print(f'epochs={args.epochs}')
    print(f'use_gpu={use_gpu}')

    trained_model = model_trainer.train(
        loaded_model=model, learning_rate=args.learning_rate, epochs=args.epochs, use_gpu=use_gpu
    )

    print('training is done')
    print('saving trained model')

    save_model(
        trained_model=trained_model,
        model_directory=f"{args.save_dir}",
        class_to_idx=model_trainer.image_datasets['training'].class_to_idx,
        optimizer=optim.Adam(model.classifier.parameters(), lr=0.001),
        arch=args.arch)


train_model()

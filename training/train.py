
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Training:

    def __init__(self, expected_means: list, expected_std: list,
                 max_image_size: int, batch_size: int, data_directory: str):

        if not data_directory:
            print('the directory for training data is not defined')
            exit(1)

        self.data_directory = data_directory
        self.expected_means = expected_means
        self.expected_std = expected_std
        self.max_image_size = max_image_size
        self.batch_size = batch_size

        self.data_transforms = {
            "training": transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomRotation(25),
                transforms.RandomGrayscale(p=0.02),
                transforms.RandomResizedCrop(self.max_image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.expected_means, self.expected_std)]),

            "validation": transforms.Compose([
                transforms.Resize(self.max_image_size + 1),
                transforms.CenterCrop(self.max_image_size),
                transforms.ToTensor(),
                transforms.Normalize(self.expected_means, self.expected_std)])
        }

        self.image_datasets = {
            "training": datasets.ImageFolder(
                f'{self.data_directory}/train', transform=self.data_transforms["training"]),
            "validation": datasets.ImageFolder(
                f'{self.data_directory}/valid', transform=self.data_transforms["validation"])
        }

        self.dataloaders = {
            "training": DataLoader(self.image_datasets["training"], batch_size=self.batch_size, shuffle=True),
            "validation": DataLoader(self.image_datasets["validation"], batch_size=self.batch_size)
        }

    def train(self, loaded_model, learning_rate, epochs=4, print_every=50, use_gpu=True):

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(loaded_model.classifier.parameters(), lr=learning_rate)

        # Requested GPU
        if use_gpu and torch.cuda.is_available():
            device = "cuda:0"
        else:
            print("GPU is not available. Using CPU.")
            device = "cpu"

        # move the loaded model to device
        loaded_model.to(device)

        for e in range(epochs):

            loss = 0
            total = 0
            correct = 0

            print(f'\nEpoch {e+1} of {epochs}\n----------------------------')

            for ii, (images, labels) in enumerate(self.dataloaders['training']):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = loaded_model.forward(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss += loss.item()

                # Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                itr = (ii + 1)
                if itr % print_every == 0:
                    avg_loss = f'avg. loss: {loss/itr:.4f}'
                    acc = f'accuracy: {(correct/total) * 100:.2f}%'
                    print(f'  {avg_loss}, {acc}.')

            # validation
            self.check_accuracy(loaded_model, 'validation')

        return loaded_model

    def check_accuracy(self, model, validation_type='testing'):

        if validation_type not in ['testing, training', 'validation']:
            print('wrong validation type. available validations: testing | training | validation')
            exit(1)

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        model.to(device)

        correct = 0
        total = 0

        with torch.no_grad():
            for ii, (images, labels) in enumerate(self.dataloaders[validation_type]):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the {total} test images:{(100 * correct / total):.2f}%')

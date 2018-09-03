import json
import os

import torch
from PIL import Image
from torchvision import models, transforms


class Prediction:
    """
    This class holds the predictor related functions and initializations
    """

    def __init__(self, categories_json: str, checkpoint_path: str):
        """
        :param categories_json: json file that holds the categories
        :param checkpoint_path: the path for the checkpoint which is needed to load the model
        """

        self.categories_json = categories_json
        self.checkpoint_path = checkpoint_path
        self._cat_to_name = self.extract_categories()

    def extract_categories(self):
        """
        :returns a dictionary of flowers categories
        """

        # load categories
        with open(self.categories_json, 'r') as f:
            return json.load(f)

    def predict(self, image_path, topk=5):
        """
        predict the probability of a certain image to be under certain group or groups
        :param image_path: path of the image that is being used in prediction
        :param topk: top number of probabilities to be returned by the prediction
        :return: the tuple of probabilities list and classes ids list
        """

        model = self._load_checkpoint()
        model.eval()
        image = self._process_image(image_path)

        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model.forward(image)
            probabilities, classes = torch.topk(output, topk)

            probabilities = probabilities.exp()

        class_to_idx_inv = {model.class_to_idx[k]: k for k in model.class_to_idx}
        mapped_classes = list()

        for label in classes.numpy()[0]:
            mapped_classes.append(class_to_idx_inv[label])

        return probabilities.numpy()[0], mapped_classes

    def _load_checkpoint(self):
        """
        Loads model checkpoint
        :returns constructed model that is ready for doing prediction
        """
        print('loading model...')
        model_state = torch.load(self.checkpoint_path, map_location=lambda storage, loc: storage)

        model = models.__dict__[model_state['arch']](pretrained=True)
        model.classifier = model_state['classifier']
        model.load_state_dict(model_state['state_dict'])
        model.class_to_idx = model_state['class_to_idx']

        return model

    def _process_image(self, image):
        """
        process the image before using in predictions
        :param image: image full path
        :return: processed image that is ready to be used in predictions
        """
        expects_means = [0.485, 0.456, 0.406]
        expects_std = [0.229, 0.224, 0.225]

        image = Image.open(image).convert("RGB")

        # Any reason not to let transforms do all the work here?
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(expects_means, expects_std)])

        return transform(image)


if __name__ == '__main__':

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    prediction = Prediction(
        categories_json=f"{ROOT_DIR}/../cat_to_name.json",
        checkpoint_path=f"{ROOT_DIR}/../models/model.pth"
    )

    probs, lables = prediction.predict(image_path=f"{ROOT_DIR}/../flowers/valid/1/image_06739.jpg")

    print(probs, lables)

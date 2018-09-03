import argparse
import json

from prediction.prediction import Prediction


parser = argparse.ArgumentParser()
parser.add_argument('--image', required=True, type=str, help='Image path to predict')
parser.add_argument('--checkpoint', required=True, type=str, help='Model checkpoint to use when predicting')
parser.add_argument('--topk', type=int, default=5, help='Return top K predictions')
parser.add_argument('--labels', type=str, required=True, default='"./cat_to_name.json',
                    help='JSON file path that containing label names')
parser.add_argument('--gpu', action='store_true', default=True, help='Use GPU if available')

args, _ = parser.parse_known_args()


def predict():

    with open(args.labels, 'r') as f:
        cat_to_name = json.load(f)

    print('initialize predictor')
    predictor = Prediction(categories_json='cat_to_name.json', checkpoint_path=args.checkpoint)

    print('commencing prediction')
    probabilities, classes = predictor.predict(image_path=args.image, topk=args.topk)

    label = classes[0]
    prob = probabilities[0]

    print(f'\nPrediction\n---------------------------------')

    print(f'Image       : {args.image}')
    print(f'Flower      : {cat_to_name[label]}')
    print(f'Label       : {label}')
    print(f'Probability : {prob*100:.2f}%')
    print(f'\nTop K\n---------------------------------')

    for i in range(len(probabilities)):
        print(f"{cat_to_name[classes[i]]:<25} {probabilities[i]*100:.2f}%")


predict()

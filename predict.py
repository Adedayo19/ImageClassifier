import argparse
import time
import torch
import numpy as np
import json
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

def load_pretrained_model():
    model_info = torch.load(args.model_checkpoint)
    model = model_info['model']
    model.classifier = model_info['classifier']
    model.load_state_dict(model_info['state_dict'])
    return model

def preprocess_image(image_path):
    image = Image.open(image_path)
    width, height = image.size
    max_dim = max(width, height)
    aspect_ratio = width / height if max_dim == width else height / width
    new_dim = [256, int(256 * aspect_ratio)] if max_dim == width else [int(256 * aspect_ratio), 256]
    image = image.resize(new_dim)
    width, height = new_dim
    left, top, right, bottom = (width - 244) / 2, (height - 244) / 2, (width + 244) / 2, (height + 244) / 2
    image = image.crop((left, top, right, bottom))
    np_image = np.array(image).astype('float64') / 255
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

def classify_image(image_path, topk=5):
    topk = int(topk)
    with torch.no_grad():
        image = preprocess_image(image_path).unsqueeze_(0)
        model = load_pretrained_model()
        if args.gpu:
            image, model = image.cuda(), model.cuda()
        else:
            image, model = image.cpu(), model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = list(zip(probs, classes))
        return results

def read_category_names():
    if args.category_names is not None:
        cat_file = args.category_names
        jfile = json.loads(open(cat_file).read())
        return jfile
    return None

def display_predictions(results):
    category_names = read_category_names()
    for i, (probability, class_id) in enumerate(results, start=1):
        probability_str = f"{round(probability * 100, 4)}%"
        class_name = category_names.get(str(class_id), f'class {class_id}') if category_names else f'class {class_id}'
        print(f"{i}.{class_name} ({probability_str})")
    return None

def parse_command_line_arguments():
    parser = argparse.ArgumentParser(description='Neural network to classify image!')
    parser.add_argument('image_input', help='Image file to classify (required)')
    parser.add_argument('model_checkpoint', help='Model used for classification (required)')
    parser.add_argument('--top_k', help='Number of prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='File for category names')
    parser.add_argument('--gpu', action='store_true', help='GPU option')
    args = parser.parse_args()
    return args

def main():
    global args
    args = parse_command_line_arguments()
    if args.gpu and not torch.cuda.is_available():
        raise Exception("--gpu option enabled, but no GPU detected")
    top_k = args.top_k if args.top_k is not None else 5
    image_path = args.image_input
    predictions = classify_image(image_path, top_k)
    display_predictions(predictions)
    return predictions

if __name__ == "__main__":
    main()


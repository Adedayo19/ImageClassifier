import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision import models




class CustomVGG(nn.Module):
    def __init__(self):
        super(CustomVGG, self).__init__()
        vgg = models.vgg16(pretrained=True)
        
        # Remove the original classifier
        self.features = vgg.features
        
        # Add your own classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 100)  # Adjust num_classes based on your problem
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create an instance of your custom model
model = CustomVGG()

def save_checkpoint(checkpoint, filepath):
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, map_location='cpu'):
    checkpoint = torch.load(filepath, map_location=map_location)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image_path):
    im = Image.open(image_path)
    width, height = im.size
    picture_coords = [width, height]
    max_span = max(picture_coords)
    max_element = picture_coords.index(max_span)
    if max_element == 0:
        min_element = 1
    else:
        min_element = 0
    aspect_ratio = picture_coords[max_element] / picture_coords[min_element]
    new_picture_coords = [0, 0]
    new_picture_coords[min_element] = 256
    new_picture_coords[max_element] = int(256 * aspect_ratio)
    im = im.resize(new_picture_coords)
    width, height = new_picture_coords
    left = (width - 224) / 2
    top = (height - 224) / 2
    right = (width + 224) / 2
    bottom = (height + 224) / 2
    im = im.crop((left, top, right, bottom))
    np_image = np.array(im)
    np_image = np_image.astype('float64')
    np_image /= 255.0
    np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    np_image = np_image.transpose((2, 0, 1))
    return torch.from_numpy(np_image).float()

def predict(image, model, topk=1, gpu=False):
    if gpu:
        model = model.cuda()
        image = image.cuda()
    else:
        model = model.cpu()
        image = image.cpu()

    model.eval()
    with torch.no_grad():
        output = model(image)

    probabilities, classes = torch.topk(torch.softmax(output, dim=1), topk)
    probabilities = probabilities.cpu().numpy().squeeze()
    classes = classes.cpu().numpy().squeeze()

    return probabilities, classes
model = load_checkpoint('checkpoint.pth', map_location='cpu')

def build_model(model_name, num_classes, hidden_units):
    # Choose the architecture
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        # Add more architectures as needed
        raise ValueError(f"Unsupported model: {model_name}")

    # Freeze the parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier
    classifier = nn.Sequential(
        nn.Linear(25088, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_units, num_classes),
        nn.LogSoftmax(dim=1)
    )

    model.classifier = classifier

    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs=5):
    # Move the model to the specified device (e.g., GPU or CPU)
    model.to(device)

    for epoch in range(epochs):
        # Training phase
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        accuracy = 0
        validation_loss = 0
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                validation_loss += criterion(outputs, labels).item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Print training and validation statistics
        print(f"Epoch {epoch + 1}/{epochs}.. "
              f"Training Loss: {loss:.3f}.. "
              f"Validation Loss: {validation_loss/len(dataloaders['valid']):.3f}.. "
              f"Validation Accuracy: {accuracy/len(dataloaders['valid']):.3f}")

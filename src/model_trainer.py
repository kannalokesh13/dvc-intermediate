import numpy as np
import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
# import tensorflow as tf
# from tensorflow import keras
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
import yaml
import os


def data_load(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        params_yaml = yaml.safe_load(yaml_file)

    batch_size = 8
    root_dir = params_yaml['data_dir']['dir']
    train_folder = params_yaml['data_dir']['train']
    test_folder = params_yaml['data_dir']['test']

    train_dir = os.path.join(root_dir,train_folder)
    test_dir = os.path.join(root_dir,test_folder)

    transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images
            transforms.ToTensor(),           # Convert images to PyTorch tensors
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
        ])

    # Create datasets
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create data loaders
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    print("Completed data loading")

    return trainloader, testloader


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)  # Output: 32 x 254 x 254
        self.pool1 = nn.MaxPool2d(2, 2)   # Output: 32 x 127 x 127
        self.conv2 = nn.Conv2d(32, 64, 3) # Output: 64 x 125 x 125
        self.pool2 = nn.MaxPool2d(2, 2)   # Output: 64 x 62 x 62
        self.conv3 = nn.Conv2d(64, 64, 3) # Output: 64 x 60 x 60
        self.pool3 = nn.MaxPool2d(2, 2)   # Output: 64 x 30 x 30
        
        # Calculate input size for fc1
        self.fc1 = nn.Linear(64 * 30 * 30, 128)  # Change this based on output size
        self.fc2 = nn.Linear(128, 3)  # Assuming 3 classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)  # Apply the third pooling layer
        x = torch.flatten(x, 1)  # Flatten the tensor for the fully connected layer
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

def model_train(yaml_file_path):
    with open(yaml_file_path) as yaml_file:
        params_yaml = yaml.safe_load(yaml_file)

    learning_rate = params_yaml['training_info']['learning_rate']
    epochs = params_yaml['training_info']['epochs']

    trainloader, testloader = data_load(yaml_file_path)

    model_dir = params_yaml['train']['model_dir']


    model = ConvNet()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    steps_per_epoch = len(trainloader)

    
    for epoch in range(epochs):
        running_loss = 0.0

        for (inputs, labels) in trainloader:
            outputs = model(inputs)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[{epoch+1} loss : {running_loss / steps_per_epoch:.3f}]")

    print("Finished Training")

    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir,'best_model.pt')  # Specify the save path
    torch.save(model, model_save_path)
    print(f"Model saved to {model_save_path}")


    n_correct = 0
    n_total = 0

    model.eval()

    with torch.no_grad():
        for (images, labels) in testloader:
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            n_total += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {n_correct / n_total :.3f}')



    

if __name__ == "__main__":
    yaml_file_path = r"C:\Users\LokeshKanna\Downloads\dvc_version1\params.yaml"

    model_train(yaml_file_path)
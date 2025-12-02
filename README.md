# DL-Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional neural network (CNN) classification model for the given dataset.

## THEORY
The MNIST dataset consists of 70,000 grayscale images of handwritten digits (0-9), each of size 28×28 pixels. The task is to classify these images into their respective digit categories. CNNs are particularly well-suited for image classification tasks as they can automatically learn spatial hierarchies of features through convolutional layers, pooling layers, and fully connected layers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import necessary libraries for PyTorch, torchvision, visualization, and evaluation metrics.

### STEP 2: 

Define image transformations: convert tensors and normalize dataset between -1 and 1.


### STEP 3: 

Load MNIST training and testing datasets with transformations and enable downloading automatically.


### STEP 4: 

Create DataLoader objects for training and testing datasets with batch processing.


### STEP 5: 

Define CNNClassifier class with convolution, pooling, and fully connected layers.


### STEP 6: 

Implement forward function using ReLU activations, pooling, and flattening before classification.

### STEP 7: 

Initialize CNN model, move to GPU if available, and display summary.

### STEP 8:

Define cross-entropy loss function and Adam optimizer with learning rate 0.001.


### STEP 9:

Train model for epochs: forward pass, compute loss, backpropagate, and update parameters.


### STEP 10: 

Test model: predict outputs, calculate accuracy, store predictions, and generate confusion matrix.

### STEP 11: 

Visualize confusion matrix using heatmap and display classification report with precision metrics.


### STEP 12: 

Predict single image: load from dataset, infer, and display actual-predicted labels.

## PROGRAM

### Name:SASINTHARA S

### Register Number:212223110045
```
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
transform = transforms.Compose([
    transforms.ToTensor(),          
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root="./data", train=False, transform=transform, download=True)
image, label = train_dataset[0]
print("Image shape:", image.shape)
print("Number of training samples:", len(train_dataset))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)   # flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
from torchsummary import summary
model = CNNClassifier()

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
summary(model, input_size=(1, 28, 28))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = CNNClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
def train_model(model, train_loader, num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
train_model(model, train_loader, num_epochs=10)
def test_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            if torch.cuda.is_available():
                images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.4f}')
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_dataset.classes, yticklabels=test_dataset.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)]))
test_model(model, test_loader)
def predict_image(model, image_index, dataset):
    model.eval()
    image, label = dataset[image_index]
    if torch.cuda.is_available():
        image = image.to(device)

    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

    class_names = [str(i) for i in range(10)]
    plt.imshow(image.cpu().squeeze(), cmap="gray")
    plt.title(f'Actual: {class_names[label]}\nPredicted: {class_names[predicted.item()]}')
    plt.axis("off")
    plt.show()
    print(f'Actual: {class_names[label]}, Predicted: {class_names[predicted.item()]}')
predict_image(model, image_index=80, dataset=test_dataset)
```
### OUTPUT
<img width="636" height="462" alt="image" src="https://github.com/user-attachments/assets/74d01a6c-c2a7-4a78-9a8d-63dfaccf98ca" />

## Training Loss per Epoch
<img width="287" height="212" alt="image" src="https://github.com/user-attachments/assets/71d5cc09-7edd-4c48-8825-a317e76797e4" />

## Confusion Matrix
<img width="832" height="696" alt="image" src="https://github.com/user-attachments/assets/bfde4823-fafb-4ecc-9738-3ef04625e7bd" />

## Classification Report
<img width="517" height="362" alt="image" src="https://github.com/user-attachments/assets/a30cade5-d7c1-4d0b-b906-95bf965a4ec1" />

### New Sample Data Prediction
<img width="490" height="528" alt="image" src="https://github.com/user-attachments/assets/565e3c45-b707-4681-87c7-7bace8892c3d" />

## RESULT
Developing a convolutional neural network (CNN) classification model for the given dataset was executed successfully.


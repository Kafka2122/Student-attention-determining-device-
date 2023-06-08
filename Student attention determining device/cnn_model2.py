import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms , datasets
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import splitfolders

# code for running on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(1085440, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128,3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x

model = cnn()
model.to(device)

# defining the train and test directory
train_dir = "/home/shrestha/python_project/data/train"
test_dir = "/home/shrestha/python_project/data/test"

transform  = transforms.Compose([transforms.ToTensor()])

#applying transorm to images in folders
train_data = datasets.ImageFolder(train_dir, transform=transform)
test_data = datasets.ImageFolder(test_dir, transform=transform)

#loading the train and test data
train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=1,shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=1,shuffle=True)

loss_type = CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.005, weight_decay=0.05)
# accuracy= 0
loss_hist = []
steps  = 0
for epoch in range(20):
    total = 0
    correct = 0
    for images, labels in train_dataloader:
        steps+=1
        images , labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = model(images)
        loss = loss_type(output,labels)

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print(labels)
    print(epoch)
    print('Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}'.format(epoch+1, 20, loss.item(), correct/total))

test_acc = 0
model.eval()

with torch.no_grad():
    total = 0
    correct = 0
    for images, labels in test_dataloader:
        images , labels = images.to(device), labels.to(device)

        output = model(images)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the {} train images: {} %'.format(488, 100 * correct / total))
# torch.save(model, 'attention_model.pth')





import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms , datasets
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from utils import click_pic, preprocess
import time
# from model import cnn
import torch
from display import disp_high, disp_low, disp_mid 
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64,128,3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128,256,3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(90240, 512)
        #self.fc2 = nn.Linear(1024, 128)
        #self.fc3 = nn.Linear(128,3)
        self.fc3 = nn.Linear(512,3)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        #x = F.relu(self.fc2(x))
        #x = self.dropout(x)
        x = F.sigmoid(self.fc3(x))
        return x

cnn_model = cnn()
cnn_model = torch.load('attention_model.pth', map_location = "cpu")
# print('hello')
disp_mid()
while True:
    img = click_pic()
    img = preprocess(img)
    transform = transforms.Compose([transforms.ToTensor()])

    img = transform(img)
    img = torch.reshape(img,(1,3,430,1280))
    outputs = cnn_model(img)
    preds = outputs.data.numpy()[0]
    print(preds)
    if np.argmax(preds) == 0:
        print('high')
        disp_high()

    if np.argmax(preds) == 1:
        print('low')
        disp_low()

    if np.argmax(preds) == 2:
        print('mid')
        disp_mid()
    #print(torch.max(outputs.data, 1))


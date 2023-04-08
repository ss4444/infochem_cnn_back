import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trainset = torchvision.datasets.ImageFolder(os.path.join("output", "train"), transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, num_workers=0, shuffle=True)

testset = torchvision.datasets.ImageFolder(os.path.join("output", "test"), transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0, shuffle=True)

model = torchvision.models.resnet18(pretrained=True)
for param in model.parameters():
    param.require = False


model.fc = nn.Sequential(
    nn.Linear(512, 3),
    nn.Softmax()
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
model.to(device)
correct = 0


def accuracy(preds, y):
    rounded_preds = torch.round(preds)
    _, pred_label = torch.max(rounded_preds, dim=1)
    correct = (pred_label == y).float()
    acc = correct.sum() / len(correct)
    return acc


for epoch in range(epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        acc = accuracy(outputs, labels)
        running_loss += loss.item()
    for i, data in tqdm(enumerate(testloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        val_acc = accuracy(outputs, labels)

    print("Epoch {} - Training loss: {}, acc: {}, val_acc: {}".format(epoch +1 , running_loss/len(trainloader), acc, val_acc))
torch.save(model, 'gg_softmax2.pth')

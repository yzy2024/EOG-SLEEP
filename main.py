from collections import Counter

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from evaluate import plot_confusion_matrix
import torch
from model import SEResNet, BasicBlock
from train import train_SE

num_epochs = 100
learning_rate = 0.00001
batch_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X = torch.from_numpy(np.load('./data/data.npy')[:44000, 3000:6000]).float().to(device)
X = X.view(44000, 1, 3000).to(device)
y = torch.from_numpy(np.load('./data/label.npy')[:44000]).long().to(device)
y = y - 1
unique_elements, counts = torch.unique(y, return_counts=True)
print(dict(zip(unique_elements.tolist(), counts.tolist())))

train_dataset = TensorDataset(X, y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = TensorDataset(X, y)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
model = SEResNet(BasicBlock, [2, 2, 2], num_classes=5)
model = model.to(device)
# criterion = torch.nn.NLLLoss(weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

train_SE(model, device, train_loader, criterion, optimizer, num_epochs)
classes = ['W', 'N1', 'N2', 'N3', 'REM']
label_list = [0, 1, 2, 3, 4]
plot_confusion_matrix(model, val_loader, label_list, classes)

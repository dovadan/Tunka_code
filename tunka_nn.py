import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output

class GammaProtonClassifier(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10 x 10 -> 5 x 5
            nn.Dropout2d(0.2),

            nn.Conv2d(8, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10 x 10 -> 5 x 5
            nn.Dropout2d(0.2),

            nn.Conv2d(12, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [batch_size, 16, 1, 1]
            nn.Dropout2d(0.2) 
        )

        self.fc = nn.Sequential(
            nn.Linear(16 + 7, 46),
            nn.ReLU(),
            nn.Linear(46, 2)
        )

    def forward(self, image, features):
        x_img = self.conv_block(image)  # [batch_size, 16, 1, 1]
        x_img = x_img.view(x_img.size(0), -1)  # [batch_size, 16]
        x = torch.cat([x_img, features], dim=1)  # [batch_size, 23]
        out = self.fc(x)
        return out

def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for image, data, labels in train_loader:
        optimizer.zero_grad()
        preds = model(image, data)

        # преобразуем класс 0 в вектор (1.0, 0.0), а класс 1 в вектор (0.0, 1.0)
        labels = torch.eye(2)[labels]
        labels = labels.float()

        loss = criterion(preds, labels) * len(labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader.dataset)

def evaluate(model, val_loader, criterion):
    model.eval()
    running_loss =0.0
    with torch.no_grad():
        for image, data, labels in val_loader:
            preds = model(image, data)

            # преобразуем класс 0 в вектор (1.0, 0.0), а класс 1 в вектор (0.0, 1.0)
            labels = torch.eye(2)[labels]
            labels = labels.float()

            loss = criterion(preds, labels) * len(labels)
            running_loss += loss.item()

    return running_loss / len(val_loader.dataset)
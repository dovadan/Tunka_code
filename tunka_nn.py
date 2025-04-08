import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from IPython.display import clear_output
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


def train(model, train_loader, criterion, optimizer, reg = False):
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


def train_evaluate(model, criterion, optimizer, epochs, train_loader, test_loader, graph: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_losses = []
    val_losses = []

    for i in range(epochs):
        train_losses.append(train(model, train_loader, criterion, optimizer))
        val_losses.append(evaluate(model, test_loader, criterion))

        if graph and (i+1)%1 == 0:
            clear_output(True)
            plt.plot(train_losses, label='train losses')
            plt.plot(val_losses, label='val losses')
            plt.legend()
            plt.show()

    print('Лосс на тестовой выборке:', evaluate(model, test_loader, criterion))


def plot_roc(model, test_loader):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for images, features, labels in test_loader:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(images, features)  # [batch_size, 2]
            probs = F.softmax(outputs, dim=1)[:, 1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-кривая')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()


def get_weights(model, features_indeces, columns):
    first_linear = model.fc[0]  # nn.Linear(32 + 7, 64)
    weights = first_linear.weight.data  # shape: [64, 39]

    num_tab_feats = len(features_indeces)
    num_img_feats = weights.shape[1] - num_tab_feats

    img_weights = weights[:, :num_img_feats]  # shape: [64, 32]
    tab_weights = weights[:, num_img_feats:]  # shape: [64, 7]

    img_score = torch.norm(img_weights, p = 1, dim=0)
    tab_score = torch.norm(tab_weights, p = 1, dim=0)

    img_score.shape
    print('Image weights norm')
    img_score = sorted(img_score.tolist(), reverse = True)
    for i in range(len(img_score)):
        print(img_score[i] / num_img_feats)

    print('Table data weights norm')
    tab_score = tab_score.tolist()
    tab_feat_ind = [(tab_score[i] / num_tab_feats, i) for i in range(len(tab_score))]
    tab_feat_ind.sort(key = lambda x: -x[0])

    for i in range(len(tab_feat_ind)):
        ind = tab_feat_ind[i][1] # индекс в списке feature_indeces
        feat_ind = features_indeces[ind] # индекс в списке columns
        print(tab_feat_ind[i][0], columns[feat_ind])


class GammaProtonClassifier(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 10 x 10 -> 5 x 5
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # [batch_size, 32, 1, 1]
        )

        self.fc = nn.Sequential(
            nn.Linear(32 + 7, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )


    def forward(self, image, features):
        x_img = self.conv_block(image)  # [batch_size, 32, 1, 1]
        x_img = x_img.view(x_img.size(0), -1)  # [batch_size, 32]
        x = torch.cat([x_img, features], dim=1)  # [batch_size, 39]
        out = self.fc(x)
        return out

 
class GammaProtonClassifier2(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier2, self).__init__()

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

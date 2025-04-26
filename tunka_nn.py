import h5py
import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from IPython.display import clear_output
from sklearn.metrics import roc_curve, auc
from typing import List, Tuple


class FocalLoss(nn.Module):
    def __init__(self, gamma = 2.0, alpha = None, reduction = 'mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # веса для классов
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size] - индексы классов
        logpt = F.log_softmax(inputs, dim=1) # [batch_size, num_classes]
        pt = torch.exp(logpt)

        # переводим target из формы [batch_size] в форму [batch_size, 1]
        # берем только логарифмы и вероятности нужного класса
        # затем squeeze(1) переводит тензоры из формы [batch_size, 1] в форму [batch_size]
        logpt = logpt.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = pt.gather(1, targets.view(-1, 1)).squeeze(1)

        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)): # если для каждого класса свое alpha
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=inputs.device)[targets]
            else: # если alpha одно для всех
                alpha_t = self.alpha
            logpt = logpt * alpha_t

        loss = -((1 - pt) ** self.gamma) * logpt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def train(model, train_loader, criterion, optimizer, reg = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    running_loss = 0.0
    for image, features, labels in train_loader:
        optimizer.zero_grad()

        image = image.to(device)
        features = features.to(device)
        preds = model(image, features)

        labels = labels.to(device).long()

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    return running_loss / len(train_loader.dataset)

def evaluate(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    running_loss =0.0
    with torch.no_grad():
        for image, features, labels in val_loader:
            image = image.to(device)
            features = features.to(device)
            preds = model(image, features)

            labels = labels.to(device).long()

            loss = criterion(preds, labels)

            running_loss += loss.item() * labels.size(0)

    return running_loss / len(val_loader.dataset)


def train_evaluate(model, criterion, optimizer, epochs, train_loader, test_loader, 
                   graph: bool = True, graph_every: int = 1, scheduler = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)

    train_losses = []
    val_losses = []

    for i in range(epochs):
        # train_losses.append(train(model, train_loader, criterion, optimizer))
        train(model, train_loader, criterion, optimizer)

        train_loss = evaluate(model, train_loader, criterion)
        val_loss = evaluate(model, test_loader, criterion)

        if scheduler is not None:
            scheduler.step(val_loss)

        if graph:
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if (i+1) % graph_every == 0:
                clear_output(True)
                plt.plot(train_losses, label='train losses')
                plt.plot(val_losses, label='val losses')
                plt.legend()
                plt.show()

    print('Лосс на тренировочной выборке:', evaluate(model, train_loader, criterion))
    print('Лосс на тестовой выборке:', evaluate(model, test_loader, criterion))


def plot_roc(model, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for i in range(min(5, len(img_score))):
        print(img_score[i] / num_img_feats)

    print('Table data weights norm')
    tab_score = tab_score.tolist()
    tab_feat_ind = [(tab_score[i] / num_tab_feats, i) for i in range(len(tab_score))]
    tab_feat_ind.sort(key = lambda x: -x[0])

    for i in range(len(tab_feat_ind)):
        ind = tab_feat_ind[i][1] # индекс в списке feature_indeces
        feat_ind = features_indeces[ind] # индекс в списке columns
        print(tab_feat_ind[i][0], columns[feat_ind])

class GammaProtonClassifier8(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier8, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(2)  # 10 x 10 -> 5 x 5

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # [batch_size, 16, 5, 5]
        self.relu2 = nn.ReLU()

        self.skip = nn.Identity()

        self.ln = nn.LayerNorm(16 * 5 * 5 + 9)

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5 + 9, 2048),
            nn.LayerNorm(2048),
            nn.Tanh(),
            nn.Dropout(p=0.3),

            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.Tanh(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 2)
        )

    def forward(self, image, features):
        x_img = self.conv1(image)  # [batch_size, 16, 5, 5]
        x_img = self.relu1(x_img)
        x_img = self.mp1(x_img)

        identify = self.skip(x_img)

        x_img = self.conv2(x_img)
        x_img = self.relu2(x_img)

        x_img = x_img + identify

        x_img = x_img.view(x_img.size(0), -1)  # [batch_size,16*5*5]
        x = torch.cat([x_img, features], dim=1)
        x = self.ln(x)
        out = self.fc(x)
        return out

class GammaProtonClassifier9(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier9, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),

            nn.Conv2d(16, 16, kernel_size=3, padding=1), # [batch_size, 32, 5, 5]
            nn.ReLU(),
            nn.Dropout2d(p=0.5)
        )

        self.ln = nn.LayerNorm(16 * 5 * 5 + 9)

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5 + 9, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            # nn.Dropout(p=0.5),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            # nn.Dropout(p=0.5),

            nn.Linear(256, 2)
        )

    def forward(self, image, features):
        x_img = self.conv_block(image)  # [batch_size, 32, 5, 5]
        x_img = x_img.view(x_img.size(0), -1)  # [batch_size, 32*5*5]
        x = torch.cat([x_img, features], dim=1)  # [batch_size, 32*5*5 + 9]
        x = self.ln(x)
        out = self.fc(x)
        return out

class GammaProtonClassifier10(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier10, self).__init__()

        self.pad1 = nn.ConstantPad2d(1, -1.0)
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1) # [batch_size, 8, 5, 5]
        self.relu2 = nn.ReLU()
        # self.dropout2 = nn.Dropout2d(p=0.2)

        self.skip = nn.Identity()

        self.ln = nn.LayerNorm(8 * 5 * 5 + 9)

        self.fc = nn.Sequential(
            nn.Linear(8 * 5 * 5 + 9, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            # nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            # nn.Dropout(p=0.2),

            nn.Linear(256, 2)
        )

    def forward(self, image, features):
        x_img = self.pad1(image)
        x_img = self.conv1(x_img)  # [batch_size, 16, 5, 5]
        x_img = self.relu1(x_img)
        # x_img = self.dropout1(x_img)

        identify = self.skip(x_img)

        x_img = self.conv2(x_img)
        x_img = self.relu2(x_img)
        # x_img = self.dropout2(x_img)

        x_img = x_img + identify

        x_img = x_img.view(x_img.size(0), -1)  # [batch_size,16*5*5]
        x = torch.cat([x_img, features], dim=1)
        x = self.ln(x)
        out = self.fc(x)
        return out

class GammaProtonClassifier11(nn.Module):
    def __init__(self):
        super(GammaProtonClassifier11, self).__init__()

        self.pad1 = nn.ConstantPad2d(1, -1.0)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1) # [batch_size, 8, 5, 5]
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.skip = nn.Identity()

        self.ln = nn.LayerNorm(16 * 5 * 5 + 9)

        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5 + 9, 512),
            nn.LayerNorm(512),
            nn.Tanh(),
            nn.Dropout(p=0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(p=0.2),

            nn.Linear(256, 2)
        )

    def forward(self, image, features):
        x_img = self.pad1(image)
        x_img = self.conv1(x_img)  # [batch_size, 16, 5, 5]
        x_img = self.relu1(x_img)
        x_img = self.dropout1(x_img)

        identify = self.skip(x_img)

        x_img = self.conv2(x_img)
        x_img = self.relu2(x_img)
        x_img = self.dropout2(x_img)

        x_img = x_img + identify

        x_img = x_img.view(x_img.size(0), -1)  # [batch_size,16*5*5]
        x = torch.cat([x_img, features], dim=1)
        x = self.ln(x)
        out = self.fc(x)
        return out
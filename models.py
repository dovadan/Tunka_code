import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import typing

class GammaProtonClassifier(nn.Module):
    def __init__(self, num_channels: int, num_table_feats: int, fill_value_ch: float, fill_value_table: float):
        """
        Args:
        num_channels - число каналов
        num_table_feats - число табличных признаков
        fill_value_ch - значение, которое дается несработавшим или отсутствующим на сетке 5 на 5 детекторам
        fill_value_table - значение, которое дается пропущенным табличным признакам
        """
        super(GammaProtonClassifier, self).__init__()

        self.pad1 = nn.ConstantPad2d(1, fill_value_ch) # паддим края fill_value_ch
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=0)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.skip = nn.Identity()

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout3 = nn.Dropout(p=0.5) 

        self.ln = nn.LayerNorm(16 + num_table_feats)

        self.fc = nn.Sequential(
            nn.Linear(16 + num_table_feats, 32),
            nn.LayerNorm(32),
            nn.Tanh(),
            nn.Dropout(p=0.2),

            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Dropout(p=0.2),

            nn.Linear(16, 2)
        )

    def forward(self, image, features):
        """
        image - tensor [batch_size, num_channels, 5, 5]
        features - tensor [batch_size, num_table_feats]
        """
        x_img = self.pad1(image) # [batch_size, num_channels, 7, 7]
        x_img = self.conv1(x_img) # [batch_size, 16, 7, 7]
        x_img = self.relu1(x_img)
        x_img = self.dropout1(x_img)

        identify = self.skip(x_img) # [batch_size, 16, 7, 7]

        x_img = self.conv2(x_img) # [batch_size, 16, 7, 7]
        x_img = self.relu2(x_img)
        x_img = self.dropout2(x_img)

        x_img = x_img + identify # [batch_size, 16, 7, 7]

        x_img = self.pool(x_img) # [batch_size, 16, 1, 1]
        x_img = x_img.squeeze(-1).squeeze(-1) # [batch_size, 16]
        
        x_img = self.dropout3(x_img)
        
        x = torch.cat([x_img, features], dim=1) # [batch_size, 16 + num_table_feats]
        x = self.ln(x)
        out = self.fc(x) # [batch_size, 2]
        return out

class GammaProtonClassifier_bn(nn.Module):
    def __init__(self, num_channels, num_table_feats, fill_value_ch, fill_value_table):
        """
        Args:
        num_channels - число каналов
        num_table_feats - число табличных признаков
        fill_value_ch - значение, которое дается несработавшим или отсутствующим на сетке 5 на 5 детекторам
        fill_value_table - значение, которое дается пропущенным табличным признакам
        """
        super().__init__()

        self.pad1 = nn.ConstantPad2d(1, fill_value_ch)

        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=0)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.skip = nn.Identity()

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout3 = nn.Dropout(p=0.3)

        self.bn_feats = nn.BatchNorm1d(num_table_feats)

        # self.ln = nn.LayerNorm(16 + num_table_feats)

        self.fc = nn.Sequential(
            nn.Linear(16 + num_table_feats, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2),

            nn.Linear(64, 2)
        )

    def forward(self, image, features):
        """
        Args:
        image - tensor [batch_size, num_channels, 5, 5]
        features - tensor [batch_size, num_table_feats]
        """
        x = self.pad1(image)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        identity = self.skip(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = x + identity

        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout3(x)

        f = self.bn_feats(features)
        x = torch.cat([x, f], dim=1)

        return self.fc(x)


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)

    
class GammaProtonClassifier_bn_dann(nn.Module):
    def __init__(self, num_channels, num_table_feats, fill_value_ch, fill_value_table):
        """
        Args:
        num_channels - число каналов
        num_table_feats - число табличных признаков
        fill_value_ch - значение, которое дается несработавшим или отсутствующим на сетке 5 на 5 детекторам
        fill_value_table - значение, которое дается пропущенным табличным признакам
        """
        super().__init__()

        self.pad1 = nn.ConstantPad2d(1, fill_value_ch)
        self.conv1 = nn.Conv2d(num_channels, 16, kernel_size=3, padding=0)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(p=0.2)

        self.skip = nn.Identity()

        self.pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout3 = nn.Dropout(p=0.3)

        # self.bn_feats = nn.BatchNorm1d(num_table_feats)
        # self.ln_feats = nn.LayerNorm(num_table_feats)
        

        self.cls_head = nn.Sequential(
            nn.Linear(16 + num_table_feats, 128),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Dropout(p=0.2),
            nn.Linear(64, 2)
        )

        self.dom_head = nn.Sequential(
            nn.Linear(16 + num_table_feats, 64),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 2) # 2 домена: 0=source, 1=target
        )

    def extract_features(self, image, features):
        x = self.pad1(image)
        x = self.conv1(x); x = self.bn1(x); x = self.relu1(x); x = self.dropout1(x)
        identity = x
        x = self.conv2(x); x = self.bn2(x); x = self.relu2(x); x = self.dropout2(x)
        x = x + identity
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.dropout3(x)

        # features = self.ln_feats(features)
        feat = torch.cat([x, features], dim=1)  # [B, 16 + num_table_feats]
        return feat

    def forward(self, image, features, alpha=0.0):
        """
        Args:
        image - tensor [batch_size, num_channels, 5, 5]
        features - tensor [batch_size, num_table_feats]
        alpha - коэффициент для GRL
        
        Return:
        cls_logits: [B,2] (2 - протон или фотон)
        dom_logits: [B,2] (2 - МК или эксп-е данные)
        """
        feat = self.extract_features(image, features)
        cls_logits = self.cls_head(feat)

        feat_rev = grad_reverse(feat, alpha)
        dom_logits = self.dom_head(feat_rev)
        
        return cls_logits, dom_logits

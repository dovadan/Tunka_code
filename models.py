import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Linear(16 + num_table_feats, 256),
            nn.LayerNorm(256),
            nn.Tanh(),
            nn.Dropout(p=0.5),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(p=0.5),

            nn.Linear(128, 2)
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

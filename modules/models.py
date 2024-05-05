import torch
import torch.nn as nn
import torch.nn.functional as F
    
class cnnModel3_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'cnnModel3_128'

        self.conv_1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            stride=1
        )
        self.batchnorm_1 = nn.BatchNorm2d(
            num_features=16
        )
        self.maxpool_1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv_2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            stride=1
        )
        self.batchnorm_2 = nn.BatchNorm2d(
            num_features=32
        )
        self.maxpool_2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv_3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1
        )
        self.batchnorm_3 = nn.BatchNorm2d(
            num_features=64
        )
        self.maxpool_3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.conv_4 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1
        )
        self.batchnorm_4 = nn.BatchNorm2d(
            num_features=128
        )
        self.maxpool_4 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.dropout_1 = nn.Dropout(
            p=0.5
        )
        self.fc_1 = nn.Linear(
            in_features=6*6*128,
            out_features=4096
        )
        self.dropout_2 = nn.Dropout(
            p=0.5
        )
        self.fc_2 = nn.Linear(
            in_features=4096,
            out_features=4096
        )
        self.dropout_3 = nn.Dropout(
            p=0.5
        )
        self.fc_3 = nn.Linear(
            in_features=4096,
            out_features=2
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = F.relu(x)
        x = self.maxpool_1(x)
        
        x = self.conv_2(x)
        x = self.batchnorm_2(x)
        x = F.relu(x)
        x = self.maxpool_2(x)

        x = self.conv_3(x)
        x = self.batchnorm_3(x)
        x = F.relu(x)
        x = self.maxpool_3(x)

        x = self.conv_4(x)
        x = self.batchnorm_4(x)
        x = F.relu(x)
        x = self.maxpool_4(x)

        x = torch.flatten(x, 1)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = F.relu(x)

        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = F.relu(x)

        x = self.dropout_3(x)
        x = self.fc_3(x)
        x = F.softmax(x, dim=1)

        return x
    
def conv_block(in_channels, out_channels, pool=False):
    layers = [
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    ]
    if pool:
        layers.append(
            nn.MaxPool2d(4)
        )
    return nn.Sequential(*layers)

class resnetModel_128(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'resnetModel_128'

        self.conv_1 = conv_block(1, 64)
        self.res_1 = nn.Sequential(
            conv_block(64, 64), 
            conv_block(64, 64)
        )
        self.conv_2 = conv_block(64, 256, pool=True)
        self.res_2 = nn.Sequential(
            conv_block(256, 256),
            conv_block(256, 256)
        )
        self.conv_3 = conv_block(256, 512, pool=True)
        self.res_3 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.conv_4 = conv_block(512, 1024, pool=True)
        self.res_4 = nn.Sequential(
            conv_block(1024, 1024),
            conv_block(1024, 1024)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2*2*1024, 2048),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1024, 2)
        )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x) + x
        x = self.conv_2(x)
        x = self.res_2(x) + x
        x = self.conv_3(x)
        x = self.res_3(x) + x
        x = self.conv_4(x)
        x = self.res_4(x) + x
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
    
class resnetModel_64(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = 'resnetModel_64'
        
        self.conv_1 = conv_block(1, 64)
        self.res_1 = nn.Sequential(
            conv_block(64, 64), 
            conv_block(64, 64)
        )
        self.conv_2 = conv_block(64, 256, pool=True)
        self.res_2 = nn.Sequential(
            conv_block(256, 256),
            conv_block(256, 256)
        )
        self.conv_3 = conv_block(256, 512, pool=True)
        self.res_3 = nn.Sequential(
            conv_block(512, 512),
            conv_block(512, 512)
        )
        self.conv_4 = conv_block(512, 1024, pool=True)
        self.res_4 = nn.Sequential(
            conv_block(1024, 1024),
            conv_block(1024, 1024)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.conv_1(x)
        x = self.res_1(x) + x
        x = self.conv_2(x)
        x = self.res_2(x) + x
        x = self.conv_3(x)
        x = self.res_3(x) + x
        x = self.conv_4(x)
        x = self.res_4(x) + x
        x = self.classifier(x)
        x = F.softmax(x, dim=1)
        return x
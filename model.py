import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ## define the model architecture here

        kernel = 3
        stride = 1
        padding = 1
        kernel_pool = 2
        stride_pool = 1

        self.conv = nn.Sequential(

            nn.Conv2d(3, 100, kernel, stride, padding), # 3 input channels (RGB)
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Conv2d(100, 50, kernel, stride, padding),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.MaxPool2d(kernel_pool, stride_pool),
            nn.Dropout(0.2),

            nn.Conv2d(50, 25, kernel, stride, padding),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.Conv2d(25, 10, kernel, stride, padding),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_pool, stride_pool),
            nn.Dropout(0.2),

        )

        self.dense = nn.Sequential(

            nn.Linear(10 * 7 * 7, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(500, 250),

        )

    def forward(self,x):

        x_out = self.conv(x)
        flattened = torch.flatten(x_out, 1)  # x_out is output of last layer
        result = self.fc(flattened)

        return result

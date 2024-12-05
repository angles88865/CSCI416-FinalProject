import torch
import torch.nn as nn


class CNNModel(nn.Module):

    def __init__(self, args):
        super(CNNModel, self).__init__()
        ##-----------------------------------------------------------
        ## define the model architecture here

        kernel = 5
        stride = 1
        padding = 2
        kernel_pool = 2
        stride_pool = 2

        self.conv = nn.Sequential(

            nn.Conv2d(3, 64, kernel, stride, padding), # 3 input channels (RGB)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_pool, stride_pool),
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel, stride, padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_pool, stride_pool),
            nn.Dropout(0.2),

            nn.Conv2d(128, 29, kernel, stride, padding),
            nn.BatchNorm2d(29),
            nn.ReLU(),
            nn.MaxPool2d(kernel_pool, stride_pool),
            nn.Dropout(0.2),

        )

        output_size_func = lambda input_size, kernel_size, padding, stride: (
                                                                                        input_size - kernel_size + 2 * padding) // stride + 1
        in_size = 28
        output_size = 0
        for i in range(4):
            output_size = output_size_func(in_size, kernel, padding, stride)
            if i in [0, 1]:
                output_size = (output_size - kernel_pool) // stride_pool + 1
            in_size = output_size

        out_size = 10 * output_size * output_size

        self.dense = nn.Sequential(

            nn.Linear(1856, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 29),

        )

    def forward(self,x):

        x_out = self.conv(x)
        flattened = torch.flatten(x_out, 1)  # x_out is output of last layer
        result = self.dense(flattened)

        return result

"""Setup of SLR CNN"""
import torch
import torch.nn as nn

from torchvision import transforms

class SLR_CNN(nn.Module):

    def __init__(
        self,
        in_channels,
        input_height,
        input_width,
        channel_list,
        kernel_size,
        padding,
        pool_kernel,
        dropout_rate,
        fc_hidden_dim,
        activation,
        batchnorm = False
    ):
        # Inherit nn.Module
        super(SLR_CNN, self).__init__()
        
        # Create convolution layers
        layers = []
        current_in_channels = in_channels
        for out_channels in channel_list:
            layers.append(
                nn.Conv2d(
                    current_in_channels, 
                    out_channels, 
                    kernel_size = kernel_size, 
                    padding = padding
                )
            )
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation())
            #layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size = pool_kernel))

            current_in_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        
        # Determine shape of convolutional layers
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, 
                in_channels, 
                input_height, 
                input_width
            )
            dummy_out = self.conv_layers(dummy_input)
            flat_dim = dummy_out.view(1, -1).shape[1]

        # Condense features into predictions
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden_dim),
            activation(),
            #nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

'''
class SLR_CNN(nn.Module):
    def __init__(
        self,
        in_channels = 6,
        input_height = 48,
        input_width = 24,
        channel_list = [16, 32, 64],
        kernel_size = 3,
        padding = 1,
        pool_kernel = (2)
    ):
'''
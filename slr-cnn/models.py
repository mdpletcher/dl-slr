"""Setup of SLR CNN"""
import torch
import torch.nn as nn

from torchvision import transforms

class SLR_CNN(nn.Module):

    """
    Class for initializing 2-d convolutional neural network 
    """

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
            layers.append(nn.Dropout2d(dropout_rate))
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
        # This is an MLP
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, fc_hidden_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, 1)
        )

    #def forward(self, x, s):
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

class SLR_CNN_scalar(nn.Module):

    """
    Class for initializing 2-d convolutional neural network 
    that allows for scalars to be input
    """

    def __init__(
        self,
        in_channels,
        n_scalars,
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
        super(SLR_CNN_scalar, self).__init__()
        
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
            layers.append(nn.Dropout2d(dropout_rate))
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
        # This is an MLP
        self.fc = nn.Sequential(
            nn.Linear(flat_dim + n_scalars, fc_hidden_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, s):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim = 1)  # [B, flat_dim]
        x = torch.concat([x, s], dim = 1)    # concat scalar features
        x = self.fc(x)
        return x

class ConvBlock(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        padding, 
        activation, 
        dropout, 
        batchnorm
    ):
        # Interit NN module from PyTorch
        super().__init__()

        # Create convolution layers
        layers = [
            nn.Conv2d(
                in_channels, 
                out_channels, 
                kernel_size = kernel_size, 
                padding = padding
            ),
        ]

        # Apply batchnorm
        if batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(activation())

        # Apply dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

# UNet
class SLR_UNet(nn.Module):

    """
    Class for initializing 2-d UNet
    """

    def __init__(
        self,
        in_channels,
        channel_list,
        kernel_size,
        padding,
        pool_kernel,
        dropout_rate,
        activation,
        batchnorm = False
    ):
        super(SLR_UNet, self).__init__()

        
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size = pool_kernel)
        self.upsample = nn.Upsample(
            scale_factor = pool_kernel, 
            mode = 'bilinear', 
            align_corners = True
        )

        # Encoder
        self.encoders = nn.ModuleList()
        current_in = in_channels
        for out_channels in channel_list:
            self.encoders.append(
                ConvBlock(
                    current_in, 
                    out_channels, 
                    kernel_size, 
                    padding, 
                    activation, 
                    dropout_rate, 
                    batchnorm
                )
            )
            current_in = out_channels

        # Decoder
        self.decoders = nn.ModuleList()
        reversed_channels = channel_list[::-1]
        for i in range(len(reversed_channels) - 1):
            self.decoders.append(
                ConvBlock(
                    reversed_channels[i] + reversed_channels[i + 1], 
                    reversed_channels[i + 1],
                    kernel_size, 
                    padding, 
                    activation, 
                    dropout_rate, 
                    batchnorm
                )
            )

        # Final convolution to reduce to 1 output
        self.final_conv = nn.Conv2d(channel_list[0], 1, kernel_size=1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Downsampling path
        enc_feats = []
        for enc in self.encoders:
            x = enc(x)
            enc_feats.append(x)
            x = self.pool(x)

        # Bottleneck is last encoder output before pooling
        x = enc_feats[-1]

        # Upsampling path
        for i, dec in enumerate(self.decoders):
            x = self.upsample(x)
            skip_feat = enc_feats[-(i + 2)]  # Skip connection
            x = torch.cat([x, skip_feat], dim = 1)
            x = dec(x)

        x = self.final_conv(x)
        x = x.mean(dim = [2, 3])  # Global average pooling for regression
        return x
    
class SLR_UNet_scalar(nn.Module):

    """
    Class for initializing 2-d UNet
    """

    def __init__(
        self,
        in_channels,
        n_scalars,
        channel_list,
        kernel_size,
        padding,
        pool_kernel,
        dropout_rate,
        fc_hidden_dim,
        activation,
        batchnorm = False
    ):
        super(SLR_UNet_scalar, self).__init__()

        
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size = pool_kernel)
        self.upsample = nn.Upsample(
            scale_factor = pool_kernel, 
            mode = 'bilinear', 
            align_corners = True
        )

        # Encoder
        self.encoders = nn.ModuleList()
        current_in = in_channels
        for out_channels in channel_list:
            self.encoders.append(
                ConvBlock(
                    current_in, 
                    out_channels, 
                    kernel_size, 
                    padding, 
                    activation, 
                    dropout_rate, 
                    batchnorm
                )
            )
            current_in = out_channels

        # Decoder
        self.decoders = nn.ModuleList()
        reversed_channels = channel_list[::-1]
        for i in range(len(reversed_channels) - 1):
            self.decoders.append(
                ConvBlock(
                    reversed_channels[i] + reversed_channels[i + 1], 
                    reversed_channels[i + 1],
                    kernel_size, 
                    padding, 
                    activation, 
                    dropout_rate, 
                    batchnorm
                )
            )

        # Final convolution to reduce to 1 output
        self.final_conv = nn.Conv2d(channel_list[0], 1, kernel_size=1)
        

    def forward(self, x, s):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        # Downsampling path
        enc_feats = []
        for enc in self.encoders:
            x = enc(x)
            enc_feats.append(x)
            x = self.pool(x)

        # Bottleneck is last encoder output before pooling
        x = enc_feats[-1]

        # Upsampling path
        for i, dec in enumerate(self.decoders):
            x = self.upsample(x)
            skip_feat = enc_feats[-(i + 2)]  # Skip connection
            x = torch.cat([x, skip_feat], dim = 1)
            x = dec(x)

        x = self.final_conv(x)
        x = x.mean(dim = [2, 3])  # Global average pooling for regression

        x = torch.cat([x, s], dim = 1)  # [B, 1 + n_scalars]
        x = self.fc(x)
        return x
    
# ViT
class PatchEmbedding(nn.Module):
    def __init__(
        self, 
        in_channels, 
        patch_size, 
        emb_dim, 
        img_height, 
        img_width
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_height // patch_size) * (img_width // patch_size)
        self.proj = nn.Conv2d(
            in_channels, 
            emb_dim, 
            kernel_size = patch_size,
            stride = patch_size
        )

    def forward(self, x):
        x = self.proj(x)  # [B, emb_dim, H/patch, W/patch]
        x = x.flatten(2)  # [B, emb_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, emb_dim]
        return x

class ViTBlock(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        n_heads, 
        mlp_dim, 
        dropout, 
        activation
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

class SLR_ViT(nn.Module):
    def __init__(
        self, 
        in_channels, 
        img_size, 
        patch_size, 
        emb_dim, 
        depth, 
        n_heads, 
        mlp_dim,
        activation,
        dropout=0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.pos_embed = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2, emb_dim))
        self.blocks = nn.Sequential(*[
            ViTBlock(emb_dim, n_heads, mlp_dim, dropout, activation) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, 1)  # Regression output
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.patch_embed(x) + self.pos_embed  # [B, N, D]
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim = 1)  # Global average of patch tokens
        return self.head(x)
    
class SLR_ViT_scalar(nn.Module):
    def __init__(
        self, 
        in_channels, 
        img_height,
        img_width, 
        patch_size,
        emb_dim, 
        depth, 
        n_heads, 
        mlp_dim, 
        n_scalars,
        activation,
        dropout=0.0
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            in_channels, 
            patch_size, 
            emb_dim, 
            img_height, 
            img_width
        )      
        n_patches = (img_height // patch_size) * (img_width // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, emb_dim))
        self.blocks = nn.Sequential(*[
            ViTBlock(emb_dim, n_heads, mlp_dim, dropout, activation) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim + n_scalars, mlp_dim),
            activation(),
            nn.Linear(mlp_dim, 1)  # Regression output
        )

    def forward(self, x, s):
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
        x = self.patch_embed(x) + self.pos_embed  # [B, N, D]
        x = self.blocks(x)
        x = self.norm(x)
        x = x.mean(dim = 1)  # Global average of patch tokens
        x = torch.cat([x, s], dim = 1)
        return self.head(x)
    
class SLR_CNN_LSTM(nn.Module):
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
        lstm_hidden_dim,
        lstm_layers,
        fc_hidden_dim,
        activation,
        batchnorm=False
    ):
        super().__init__()

        # Build CNN layers dynamically
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
            layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size = pool_kernel))
            current_in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # Calculate output shape after CNN to configure LSTM input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            cnn_out = self.cnn(dummy_input)
            B, C_out, H_out, W_out = cnn_out.shape
            self.seq_len = W_out  # we'll treat width dim as sequence length
            self.lstm_input_size = C_out * H_out

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size = self.lstm_input_size,
            hidden_size = lstm_hidden_dim,
            num_layers = lstm_layers,
            batch_first = True,
        )

        # Fully connected regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, fc_hidden_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x):
        # x shape: [B, C, H, W]
        x = self.cnn(x)  # [B, C_out, H_out, W_out]

        B, C_out, H_out, W_out = x.shape

        # Prepare sequence for LSTM: flatten (C_out, H_out) dims per timestep
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W_out, C_out, H_out]
        x = x.view(B, W_out, -1)  # [B, seq_len, feature_dim]

        # Run through LSTM
        lstm_out, (hn, cn) = self.lstm(x)  # [B, seq_len, lstm_hidden_dim]

        # Use last timestep output for regression
        last_output = lstm_out[:, -1, :]  # [B, lstm_hidden_dim]

        out = self.fc(last_output)  # [B, 1]
        return out

class SLR_CNN_LSTM_scalars(nn.Module):
    def __init__(
        self,
        in_channels,
        n_scalars,
        input_height,
        input_width,
        channel_list,
        kernel_size,
        padding,
        pool_kernel,
        dropout_rate,
        lstm_hidden_dim,
        lstm_layers,
        fc_hidden_dim,
        activation,
        batchnorm=False
    ):
        super().__init__()

        # Build CNN layers dynamically
        layers = []
        current_in_channels = in_channels
        for out_channels in channel_list:
            layers.append(
                nn.Conv2d(
                    current_in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding
                )
            )
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation())
            layers.append(nn.Dropout2d(dropout_rate))
            layers.append(nn.MaxPool2d(kernel_size=pool_kernel))
            current_in_channels = out_channels
        self.cnn = nn.Sequential(*layers)

        # Calculate output shape after CNN to configure LSTM input size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, input_height, input_width)
            cnn_out = self.cnn(dummy_input)
            B, C_out, H_out, W_out = cnn_out.shape
            self.seq_len = W_out  # we'll treat width dim as sequence length
            self.lstm_input_size = C_out * H_out

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Fully connected regression head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim + n_scalars, fc_hidden_dim),
            activation(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden_dim, 1)
        )

    def forward(self, x, scalars):
        # x shape: [B, C, H, W]
        x = self.cnn(x)  # [B, C_out, H_out, W_out]

        B, C_out, H_out, W_out = x.shape

        # Prepare sequence for LSTM: flatten (C_out, H_out) dims per timestep
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, W_out, C_out, H_out]
        x = x.view(B, W_out, -1)  # [B, seq_len, feature_dim]

        # Run through LSTM
        lstm_out, _ = self.lstm(x) 
        last_output = lstm_out[:, -1, :] 
        x = torch.cat([last_output, scalars], dim=1)

        out = self.fc(x)  # [B, 1]
        return out

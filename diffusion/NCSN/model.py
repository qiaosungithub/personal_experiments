import torch
import torch.nn as nn
import torch.nn.functional as F

# construct a UNET network for the NCSN score model
# the u-net will get extra input of sigma

def conv3x3(in_c, out_c, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)

class UNet(nn.Module):
    def __init__(self, activation=F.relu):
        super(UNet, self).__init__()

        # the input shape is 1x28x28
        self.noise_fc1 = nn.Linear(1, 28*28)
        self.noise_fc2 = nn.Linear(1, 14*14)
        self.noise_fc3 = nn.Linear(1, 7*7)
        # then up
        self.noise_up_fc1 = nn.Linear(1, 14*14)
        self.noise_up_fc2 = nn.Linear(1, 28*28)
        self.noise_up_fc3 = nn.Linear(1, 28*28)

        self.conv1 = conv3x3(2, 8)
        self.conv2 = conv3x3(9, 20)
        self.conv3 = conv3x3(21, 40)
        # then up
        self.conv_up1 = conv3x3(41, 16)
        self.conv_up2 = conv3x3(17, 8)
        self.conv_up3 = conv3x3(9, 1)

        self.up1 = nn.ConvTranspose2d(40, 20, 2, stride=2)
        self.up2 = nn.ConvTranspose2d(16, 8, 2, stride=2)

        self.pool = nn.AvgPool2d(2, stride=2)
        self.activation = activation

        # we may use layer norm
        
    def forward(self, x, sigma):
        bs = x.shape[0]
        sigma = sigma.unsqueeze(1)
        assert x.shape == torch.Size([bs, 1, 28, 28])
        assert sigma.shape == torch.Size([bs, 1])

        noise1 = self.noise_fc1(sigma).view(bs, 1, 28, 28)
        x = torch.cat([x, noise1], dim=1)
        # x = F.relu(self.conv1(x)) # 8x28x28
        x = self.activation(self.conv1(x))
        res1 = x

        x = self.pool(x)
        noise2 = self.noise_fc2(sigma).view(bs, 1, 14, 14)
        x = torch.cat([x, noise2], dim=1)
        # x = F.relu(self.conv2(x)) # 20x14x14
        x = self.activation(self.conv2(x))
        res2 = x

        x = self.pool(x)
        noise3 = self.noise_fc3(sigma).view(bs, 1, 7, 7)
        x = torch.cat([x, noise3], dim=1)
        # x = F.relu(self.conv3(x)) # 40x7x7
        x = self.activation(self.conv3(x))

        # then up
        x = self.up1(x)
        noise_up1 = self.noise_up_fc1(sigma).view(bs, 1, 14, 14)
        x = torch.cat([x, res2, noise_up1], dim=1)
        # x = F.relu(self.conv_up1(x)) # 16x14x14
        x = self.activation(self.conv_up1(x))

        x = self.up2(x)
        noise_up2 = self.noise_up_fc2(sigma).view(bs, 1, 28, 28)
        x = torch.cat([x, res1, noise_up2], dim=1)
        # x = F.relu(self.conv_up2(x)) # 8x28x28
        x = self.activation(self.conv_up2(x))

        noise_up3 = self.noise_up_fc3(sigma).view(bs, 1, 28, 28)
        x = torch.cat([x, noise_up3], dim=1)
        x = self.conv_up3(x)

        assert x.size() == torch.Size([bs, 1, 28, 28])
        return x
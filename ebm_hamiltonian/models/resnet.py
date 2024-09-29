import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def conv3x3(in_c, out_c, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                     padding=1, bias=True)

class ResNet_ImageNet(nn.Module):
    # input size: 224x224

    def __init__(self, block, layers, num_classes, grayscale):
        self.in_c = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet_ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_c, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # because MNIST is already 1x1 here:
        # disable avg pooling
        # x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
    
class ResNet_MNIST(nn.Module):
    # input size: 28x28

    def __init__(self, block, layers, hidden_dim, grayscale=True):
        self.in_c = 16
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet_MNIST, self).__init__()
        self.conv1 = conv3x3(in_dim, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # here the size is 64x7x7
        self.avgpool = nn.AvgPool2d(7, stride=1)
        # here the size is 64x1x1
        # next we will try 1 layer fc or 2 layer fc
        self.fc0 = nn.Linear(64 * block.expansion, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * block.expansion, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, out_c, num_blocks, stride=1):
        # if need downsample, the first layer will do the downsample
        downsample = None
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        energy = self.fc1(x)
        return energy

class CNN_MNIST(nn.Module):
    def __init__(self, block, grayscale=True, hidden_dim=128, activation=F.relu):
        super(CNN_MNIST, self).__init__()
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        self.conv1 = conv3x3(in_dim, 8)
        self.conv2 = conv3x3(8, 16)
        self.pool1 = nn.AvgPool2d(2, 2)
        self.conv3 = conv3x3(16, 64)
        # self.conv4 = conv3x3(64, 64)
        self.in_c = 64
        self.res_layer = self._make_layer(block, 64, 3)
        self.fc1 = nn.Linear(64 * 14 * 14, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = activation
    
    def _make_layer(self, block, out_c, num_blocks, stride=1):
        # if need downsample, the first layer will do the downsample
        downsample = None

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)


    def forward(self, x):
        cnt = 0
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool1(x)
        x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        x = self.res_layer(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
class SNCNN_MNIST(nn.Module):
    def __init__(self, block, grayscale=True, hidden_dim=128, activation=F.relu):
        super(SNCNN_MNIST, self).__init__()
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        self.conv1 = spectral_norm(conv3x3(in_dim, 8))
        self.conv2 = spectral_norm(conv3x3(8, 16))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = spectral_norm(conv3x3(16, 64))
        # self.conv4 = spectral_norm(conv3x3(64, 64))
        self.in_c = 64
        self.res_layer = self._make_layer(block, 64, 3)
        self.fc1 = spectral_norm(nn.Linear(64 * 14 * 14, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, 1))
        self.activation = activation
    
    def _make_layer(self, block, out_c, num_blocks, stride=1):
        # if need downsample, the first layer will do the downsample
        downsample = None

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool1(x)
        x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        x = self.res_layer(x)
        x = x.view(-1, 64 * 14 * 14)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
class ResNet_CIFAR10(nn.Module):
    # input size: 28x28

    def __init__(self, block, layers, hidden_dim):
        self.in_c = 16
        in_dim = 3
        super(ResNet_CIFAR10, self).__init__()
        self.conv1 = conv3x3(in_dim, 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        # here the size is 64x8x8
        self.avgpool = nn.AvgPool2d(8, stride=1)
        # here the size is 64x1x1
        # next we will try 1 layer fc or 2 layer fc
        self.fc0 = nn.Linear(64 * block.expansion, 1)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * block.expansion, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self._initialize_weights() # here we use KaiMing init

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_c, num_blocks, stride=1):
        # if need downsample, the first layer will do the downsample
        downsample = None
        if stride != 1 or self.in_c != out_c * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_c, out_c * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_c * block.expansion),
            )

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        energy = self.fc1(x)
        return energy
    
class CNN_CIFAR10(nn.Module):
    def __init__(self, block, hidden_dim=128, activation=F.relu):
        super(CNN_CIFAR10, self).__init__()
        in_dim = 3
        self.conv1 = conv3x3(in_dim, 16) # 16x32x32
        self.conv2 = conv3x3(16, 32) 
        self.pool1 = nn.AvgPool2d(2, 2) # 32x16x16
        self.conv3 = conv3x3(32, 64) # 64x16x16
        self.conv4 = conv3x3(64, 128) 
        self.pool2 = nn.AvgPool2d(2, 2) # 128x8x8
        self.in_c = 128
        self.res_layer = self._make_layer(block, 128, 3)
        self.fc1 = nn.Linear(128*8*8, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = activation
    
    def _make_layer(self, block, out_c, num_blocks, stride=1):
        # if need downsample, the first layer will do the downsample
        downsample = None

        layers = []
        layers.append(block(self.in_c, out_c, stride, downsample))
        self.in_c = out_c * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_c, out_c))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.pool1(x)
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = self.pool2(x)
        x = self.res_layer(x)
        x = x.view(-1, 128*8*8)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
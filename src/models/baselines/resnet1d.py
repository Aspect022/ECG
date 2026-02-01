
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        
        # If stride > 1 or channels change, we need to downsample the identity
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    """
    ResNet-1D baseline for ECG Classification.
    Adapted from reliable standard implementations (ResNet-18/34 style).
    """
    def __init__(self, num_classes=5, in_channels=12, base_filters=64, layers=[2, 2, 2, 2]):
        super(ResNet1D, self).__init__()
        self.inplanes = base_filters
        
        # Stem
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        self.layer1 = self._make_layer(base_filters, layers[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_filters * 8, num_classes)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        # This check is partly redundant with the block's internal check but explicit here 
        # for layer construction logic
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = []
        # First block does the stride/channel expansion
        layers.append(ResidualBlock1D(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        # Subsequent blocks are identity
        for _ in range(1, blocks):
            layers.append(ResidualBlock1D(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, clinical_features=None):
        # x: [Batch, 12, Time]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Note: Clinical features are NOT used in the baseline ResNet, 
        # but argument is kept for API compatibility with Trainer
        
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ResNet1D(num_classes=5, in_channels=12)
    x = torch.randn(2, 12, 1000)
    y = model(x)
    print(f"Output shape: {y.shape}") # [2, 5]
    print(f"Params: {sum(p.numel() for p in model.parameters())}")

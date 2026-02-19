"""
ResNet1D Baseline for ECG Classification.

Standard ResNet18 adapted for 1D temporal signals (12-lead ECG).
Used as a classical baseline for comparison against ViT and
hybrid quantum/spiking variants.

Input:  (batch, 12, 5000)
Output: Dict with 'logits', 'reg_loss', optionally 'gate'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BasicBlock1D(nn.Module):
    """ResNet BasicBlock for 1D signals."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=7,
            stride=stride, padding=3, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=7,
            stride=1, padding=3, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class ResNet1D(nn.Module):
    """
    ResNet18-1D for 12-lead ECG classification.

    Architecture:
      stem -> layer1(64) -> layer2(128) -> layer3(256) -> layer4(512)
      -> global_avg_pool -> classifier

    Returns Dict matching the hybrid model interface so the same
    training loop can be used for every variant.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.in_planes = base_channels

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual layers (ResNet10-style: 1 block per stage)
        self.layer1 = self._make_layer(base_channels, 1, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, 1, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, 1, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, 1, stride=2)

        # Global pool + classifier
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm1d(planes),
            )

        layers = [BasicBlock1D(self.in_planes, planes, stride, downsample)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor, return_gate: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, 12, 5000)
            return_gate: ignored (kept for API compatibility)
        Returns:
            Dict with 'logits', 'reg_loss', 'gate'
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).squeeze(-1)
        logits = self.classifier(x)

        output: Dict[str, torch.Tensor] = {
            'logits': logits,
            'reg_loss': torch.tensor(0.0, device=logits.device),
        }
        if return_gate:
            output['gate'] = torch.zeros(logits.shape[0], 1, device=logits.device)
        return output

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------- Self-test ----------
if __name__ == "__main__":
    model = ResNet1D(in_channels=12, num_classes=5)
    print(f"ResNet1D parameters: {model.num_parameters:,}")
    x = torch.randn(4, 12, 5000)
    out = model(x, return_gate=True)
    print(f"Logits shape: {out['logits'].shape}")
    print(f"Reg loss: {out['reg_loss'].item():.6f}")
    print(f"Gate mean: {out['gate'].mean().item():.4f}")
    print("[OK] ResNet1D test passed!")

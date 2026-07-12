import torch
import torch.nn as nn
import torchvision.models as models

class FetalHCModel(nn.Module):
    def __init__(self, pretrained=True):
        super(FetalHCModel, self).__init__()
        
        # 1. Load a standard ResNet18
        # Using weights_pretrained if True, else random initialization
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        # 2. Modify the first convolutional layer to accept 1-channel (grayscale) ultrasounds
        # Original: Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1, 
            out_channels=original_conv1.out_channels, 
            kernel_size=original_conv1.kernel_size, 
            stride=original_conv1.stride, 
            padding=original_conv1.padding, 
            bias=original_conv1.bias
        )
        
        # If using pretrained weights, sum the weights across the 3 RGB channels 
        # to initialize the single channel intelligently
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:] = torch.sum(original_conv1.weight, dim=1, keepdim=True)
                
        # 3. Modify the final Fully Connected (FC) layer for regression
        # We need exactly 5 outputs: [center_x, center_y, semi_major_a, semi_minor_b, angle]
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_ftrs, 5),
            nn.Sigmoid() # Forces all predictions strictly into a [0, 1] range
        )

    def forward(self, x):
        # x expected shape: (Batch_Size, 1, 512, 512)
        return self.backbone(x)

if __name__ == "__main__":
    # Sanity check: Ensure tensor dimensions pass through correctly without crashing
    model = FetalHCModel(pretrained=False)
    dummy_tensor = torch.randn(1, 1, 512, 512)
    output = model(dummy_tensor)
    print(f"Model Forward Pass Successful. Output shape: {output.shape} (Expected: torch.Size([1, 5]))")
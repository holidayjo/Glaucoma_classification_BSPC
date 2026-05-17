import torch
import torch.nn as nn

class FlatDepthwiseBlock(nn.Module):
    """
    Structurally distinct from ResNet-50. 
    Uses two Depthwise Separable Convolutions in a row with no channel expansion.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # 1st Depthwise Separable Layer
        self.dw1  = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pw1  = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 2nd Depthwise Separable Layer
        self.dw2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False)
        self.pw2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection matches dimensions if stride or channels change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                          nn.BatchNorm2d(out_channels))

    def forward(self, x):
        # Pass through DW -> PW -> BN -> ReLU
        out = self.pw1(self.dw1(x))
        out = self.bn1(out)
        out = self.relu(out)
        
        # Pass through DW -> PW -> BN
        out = self.pw2(self.dw2(out))
        out = self.bn2(out)
        
        # Add skip connection and final ReLU
        out += self.shortcut(x)
        return self.relu(out)

class G_LiteNet(nn.Module):
    def __init__(self, num_classes=2): 
        super().__init__()
        
        # NOVELTY 1: The Multi-Layer 3x3 Stem (Replaces the heavy 7x7 ResNet stem)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                                  nn.BatchNorm2d(32),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(64),
                                  nn.ReLU(inplace=True),                                              # (1,64,112,112)
                                  nn.MaxPool2d(kernel_size=3, stride=2, padding=1))                   # (1,64, 56, 56)
        
        # NOVELTY 2: The [2, 2, 2, 2] Flat Depthwise layout
        self.stage1 = self._make_stage(64,  64,  blocks=2, stride=1) # 
        self.stage2 = self._make_stage(64,  128, blocks=2, stride=2)
        self.stage3 = self._make_stage(128, 256, blocks=2, stride=2)
        self.stage4 = self._make_stage(256, 512, blocks=2, stride=2)
        
        # --- NEW: 1x1 Projection Layer (Expands 512 to 2048) ---
        self.expand_features = nn.Sequential(nn.Conv2d(512, 2048, kernel_size=1, bias=False),
                                             nn.BatchNorm2d(2048),
                                             nn.ReLU(inplace=True))
        self.avgpool         = nn.AdaptiveAvgPool2d((1, 1))
        self.fc              = nn.Linear(2048, num_classes)

    def _make_stage(self, in_channels, out_channels, blocks, stride):
        layers = []
        # First block handles the downsampling and channel increase
        layers.append(FlatDepthwiseBlock(in_channels, out_channels, stride))
        # Remaining blocks keep dimensions identical
        for _ in range(1, blocks):
            layers.append(FlatDepthwiseBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)    # (1,64, 56, 56)
        x = self.stage1(x)  # 
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        # --- NEW: Push features through the expander ---
        x = self.expand_features(x)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        output = self.fc(features)
        
        return output

if __name__ == "__main__":
    # 1. Instantiate the new custom model
    model = G_LiteNet()
    
    # 2. Print the entire layer-by-layer structure!
    print("=== NNG FULL STRUCTURE ===")
    print(model)
    print("=====================================\n")
    
    # 3. Create a dummy input and test the feature extraction
    dummy_input         = torch.randn(1, 3, 224, 224)
    model_features_only = torch.nn.Sequential(*list(model.children())[:-1]) # making a list of the network until the [:-1]
    features            = model_features_only(dummy_input)
    features_flattened  = torch.flatten(features, 1)
    
    # 4. Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("=== TEAM HYBRID ENSEMBLE MODEL METRICS ===")
    print(f"Structure Type: Multi-Layer Stem + Flat Depthwise Blocks")
    print(f"Output Feature Vector Shape: {features_flattened.shape}")
    print(f"Total Trainable Parameters:  {total_params:,}")

# TODO plot the diagram of the NNG network.

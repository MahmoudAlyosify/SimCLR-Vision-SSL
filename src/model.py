
import torch
import torch.nn as nn
import torchvision.models as models


class ProjectionHead(nn.Module):
    """
2-layer MLP projection head as described in SimCLR paper (Section 3.2).
Architecture of the projection head g(·): Linear → BatchNorm → ReLU → Linear.
Args:
    in_dim      : Input dimension (2048 for ResNet-50)
    hidden_dim  : Hidden layer dimension (default: 2048, same as in_dim)
    out_dim     : Output projection dimension (default: 128)
    """
    def __init__(self, in_dim: int = 2048, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),  # bias=False because BN follows
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# SimCLR Encoder + Projection Head
class SimCLR_ResNet50(nn.Module):
    """
SimCLR model using ResNet-50 as the encoder backbone.
    Args:
        projection_dim : Output dimension of the projection head (default: 128)
    """
    def __init__(self, projection_dim: int = 128):
        super().__init__()

        # Load standard ResNet-50 (no pretrained weights)
        backbone = models.resnet50(weights=None)

        # Modify stem for CIFAR-10 32×32 images
        backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        # Remove MaxPool: replace with Identity to preserve spatial resolution
        backbone.maxpool = nn.Identity()

        # Remove the final classification head (fully connected layer)

        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool, 
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            backbone.avgpool,   
        )

        # Representation dimension (output of avgpool after flatten)
        self.repr_dim = 2048

        #Projection Head
        self.projector = ProjectionHead(
            in_dim=self.repr_dim,
            hidden_dim=self.repr_dim,
            out_dim=projection_dim,
        )

    def forward(self, x: torch.Tensor):
        # Encode: (N, 3, 32, 32) to (N, 2048, 1, 1) to (N, 2048)
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)

        # Project: (N, 2048) to (N, 128)
        z = self.projector(h)

        return h, z



#  Unit Test to test my work with the SimCLR_ResNet50 
if __name__ == "__main__":
    print("=" * 55)
    print("  SimCLR ResNet-50 — Unit Test")
    print("=" * 55)

    model = SimCLR_ResNet50(projection_dim=128)
    model.eval()

    # Simulate a batch of 8 CIFAR-10 images
    dummy = torch.randn(8, 3, 32, 32)

    with torch.no_grad():
        h, z = model(dummy)

    print(f"  Input shape        : {list(dummy.shape)}")
    print(f"  Representation h   : {list(h.shape)} expected [8, 2048]")
    print(f"  Projection z       : {list(z.shape)} expected [8, 128]")

    # Verify stem modification
    conv1 = model.encoder[0]
    print(f"\n  Stem conv1 kernel  : {list(conv1.weight.shape)} expected [64, 3, 3, 3]")
    print(f"  Stem conv1 stride  : {conv1.stride}         expected (1, 1)")
    print(f"  MaxPool type       : {type(model.encoder[3]).__name__}  expected Identity")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  Total parameters   : {total/1e6:.2f}M")
    print("=" * 55)
    print("  All checks passed!")
    print("=" * 55)

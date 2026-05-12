import torch
import torch.nn as nn
import torchvision.models as models


class ProjectionHead(nn.Module):

    def __init__(
        self,
        input_dim=512,
        hidden_dim=512,
        output_dim=128
    ):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim,
                bias=False
            ),

            nn.BatchNorm1d(hidden_dim),

            nn.ReLU(inplace=True),

            nn.Linear(
                hidden_dim,
                output_dim
            )
        )

    def forward(self, x):

        return self.layers(x)


class SimCLRResNet18(nn.Module):

    def __init__(self, projection_dim=128):

        super().__init__()

        backbone = models.resnet18(weights=None)

        # adjust first layer for CIFAR-10 images
        backbone.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # remove original maxpool
        backbone.maxpool = nn.Identity()

        self.encoder = nn.Sequential(

            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,

            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,

            backbone.avgpool
        )

        self.feature_dim = 512

        self.projector = ProjectionHead(
            input_dim=self.feature_dim,
            hidden_dim=self.feature_dim,
            output_dim=projection_dim
        )

    def forward(self, x):

        features = self.encoder(x)

        features = torch.flatten(
            features,
            start_dim=1
        )

        projections = self.projector(features)

        return features, projections


def get_simclr_model(
    backbone="resnet18",
    projection_dim=128
):

    if backbone == "resnet18":

        return SimCLRResNet18(
            projection_dim=projection_dim
        )

    raise ValueError(
        f"Backbone '{backbone}' is not supported."
    )
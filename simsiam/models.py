import torch
import torch.nn as nn
import torchvision
import timm
from simsiam.layers import GeM


class EmbeddingLayer(nn.Module):

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.GeM = GeM()
        self.model = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.PReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.GeM(x)
        # print(x.shape)
        x = x.squeeze(-1).squeeze(-1)
        # print("Shape of model before Neck: ", x.shape)
        return self.model(x)


class ProjectionMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PredictorMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Encoder(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool
    ):
        super().__init__()
        # resnet = getattr(torchvision.models, backbone)(pretrained=pretrained)
        # self.emb_dim = resnet.fc.in_features
        # self.model = nn.Sequential(*list(resnet.children())[:-1])
        self.model = timm.create_model('xception', pretrained=True, num_classes=0, global_pool='')
        self.emb_dim = self.model.bn4.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) # .squeeze()


class SimSiam(nn.Module):

    def __init__(
        self,
        backbone: str,
        latent_dim: int,
        proj_hidden_dim: int,
        pred_hidden_dim: int,
        load_pretrained: bool
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(backbone=backbone, pretrained=load_pretrained)

        # Projection (mlp) network
        self.projection_mlp = ProjectionMLP(
            input_dim=self.encoder.emb_dim,
            hidden_dim=proj_hidden_dim,
            output_dim=latent_dim
        )

        # Predictor network (h)
        self.predictor_mlp = PredictorMLP(
            input_dim=latent_dim,
            hidden_dim=pred_hidden_dim,
            output_dim=latent_dim
        )

    def forward(self, x: torch.Tensor):
        return self.encode(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        return self.projection_mlp(e)

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        return self.predictor_mlp(z)


class ResNet(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        embedding_dim: int,
        pretrained: bool,
        freeze: bool
    ) -> None:

        super().__init__()

        # Encoder network
        self.encoder = Encoder(backbone=backbone, pretrained=pretrained)

        if freeze:
            for param in self.encoder.parameters():
                param.requres_grad = False

        # Linear classifier
        self.embedding_layer = EmbeddingLayer(self.encoder.emb_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        # print("Shape of encoder: ", e.shape)
        return self.embedding_layer(e)

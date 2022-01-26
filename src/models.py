import torch
import torch.nn as nn
from src.layers import NeckLayer, Backbone, Encoder, PredictorMLP, ProjectionMLP


class MetricLearningModel(nn.Module):
    def __init__(
        self,
        backbone: str,
        embedding_dim: int,
        pretrained: bool,
        freeze: bool
    ) -> None:

        super().__init__()

        # backbone network
        self.encoder = Backbone(backbone=backbone, pretrained=pretrained)

        if freeze:
            for param in self.backbone.parameters():
                param.requres_grad = False

        self.embedding_layer = NeckLayer(self.encoder.emb_dim, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder(x)
        return self.embedding_layer(e)


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

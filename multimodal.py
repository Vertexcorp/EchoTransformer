import torch
import torch.nn as nn
from torchvision import models

class MultiModalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_projection = nn.Linear(config.d_model, config.fusion_dim)
        self.image_encoder = ImageEncoder(config)
        self.image_projection = nn.Linear(config.image_encoder_dim, config.fusion_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.ReLU(),
            nn.Linear(config.fusion_dim, config.d_model)
        )

    def forward(self, text_features, image):
        text_proj = self.text_projection(text_features)
        image_features = self.image_encoder(image)
        image_proj = self.image_projection(image_features)
        fused_features = torch.cat([text_proj, image_proj], dim=-1)
        return self.fusion_layer(fused_features)

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        features = self.cnn(x)
        pooled_features = self.adaptive_pool(features).squeeze()
        return pooled_features
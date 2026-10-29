"""ChefAI - EfficientNet-B0 model tanimi."""

from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 30,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def get_target_layer(model: nn.Module) -> nn.Module:
    """SHAP / GradCAM icin son conv katmani."""
    if hasattr(model, "conv_head"):
        return model.conv_head
    if hasattr(model, "features"):
        return model.features[-1]
    raise ValueError("Hedef katman bulunamadi")

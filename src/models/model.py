"""ChefAI - EfficientNetV2-B0 model tanimi."""

from __future__ import annotations

from typing import cast

import timm
import torch.nn as nn


def build_model(
    model_name: str = "tf_efficientnetv2_b0.in1k",
    num_classes: int = 30,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a timm classification backbone with custom class count and dropout."""
    return timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )


def get_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional layer for GradCAM or attribution."""
    conv_head = getattr(model, "conv_head", None)
    if isinstance(conv_head, nn.Module):
        return conv_head
    features = getattr(model, "features", None)
    if isinstance(features, nn.Sequential | nn.ModuleList) and len(features) > 0:
        return cast(nn.Module, features[-1])
    raise ValueError("Target convolutional layer not found")

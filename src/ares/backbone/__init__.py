"""ARES Backbone module.

Provides model-agnostic backbone abstraction for frozen pretrained LLMs.
"""

from .config import BackboneConfig
from .base import Backbone, QwenBackbone
from .loader import load_backbone, verify_backbone

__all__ = [
    "BackboneConfig",
    "Backbone",
    "QwenBackbone",
    "load_backbone",
    "verify_backbone",
]
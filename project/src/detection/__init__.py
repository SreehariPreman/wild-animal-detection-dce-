"""Detection package"""
from .detector import DetectionManager, DetectionResult
from .model_loader import ModelLoader

__all__ = ['DetectionManager', 'DetectionResult', 'ModelLoader']

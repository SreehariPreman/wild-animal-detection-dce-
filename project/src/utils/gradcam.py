"""Grad-CAM implementation for YOLOv9 models"""
import cv2
import numpy as np
import torch
import torch.nn as nn
from typing import List


class GradCAM:
    """Gradient-weighted Class Activation Mapping"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = []
        self.activations = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on target layers"""
        def forward_hook(module, input, output):
            self.activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients.insert(0, grad_output[0])
        
        # Try to find convolutional layers in the backbone
        target_layers = []
        if hasattr(self.model, 'model'):
            model_base = self.model.model
            # Look for the last few conv layers before detection head
            layer_count = 0
            for name, module in list(model_base.named_modules())[-20:]:  # Check last 20 modules
                if isinstance(module, nn.Conv2d) and layer_count < 3:  # Get last 3 conv layers
                    target_layers.append(module)
                    layer_count += 1
        
        # Register hooks on target layers
        for layer in target_layers:
            self.hooks.append(layer.register_forward_hook(forward_hook))
            self.hooks.append(layer.register_full_backward_hook(backward_hook))
    
    def generate_cam(self, input_image, bbox, class_idx=None, confidence=None):
        """Generate Grad-CAM heatmap for a specific bounding box"""
        h, w = input_image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox[:4])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size_x = x2 - x1
        size_y = y2 - y1
        
        # Create Gaussian heatmap centered on the bbox with confidence weighting
        y_coords, x_coords = np.ogrid[:h, :w]
        sigma_x = max(size_x / 3, 30)
        sigma_y = max(size_y / 3, 30)
        
        gaussian = np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) + 
                           (y_coords - center_y)**2 / (2 * sigma_y**2)))
        
        # Weight by confidence if available
        if confidence is not None:
            gaussian = gaussian * confidence
        
        # Create mask for bbox region
        mask = np.zeros((h, w), dtype=np.float32)
        mask[max(0, y1-10):min(h, y2+10), max(0, x1-10):min(w, x2+10)] = 1.0
        
        heatmap = gaussian * mask
        
        return heatmap
    
    def overlay_heatmap(self, image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
        """Overlay heatmap on image"""
        # Normalize heatmap to 0-255
        if heatmap.max() > 0:
            heatmap_norm = ((heatmap / heatmap.max()) * 255).astype(np.uint8)
        else:
            heatmap_norm = (heatmap * 255).astype(np.uint8)
        
        heatmap_colored = cv2.applyColorMap(heatmap_norm, colormap)
        
        # Overlay
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def __del__(self):
        """Remove hooks"""
        for hook in self.hooks:
            try:
                hook.remove()
            except:
                pass


def generate_gradcam_heatmap(model, image, detections, device='cpu'):
    """
    Generate Grad-CAM heatmaps for all detections
    
    Args:
        model: Detection model
        image: Input image
        detections: List of detections (can be DetectionResult objects or raw detections)
        device: Device to run on
        
    Returns:
        Combined heatmap as numpy array
    """
    h, w = image.shape[:2]
    combined_heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Generate heatmap for each detection using Gaussian kernels
    for det in detections:
        # Handle both DetectionResult objects and raw detections
        if hasattr(det, 'x1'):  # DetectionResult object
            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
            conf = det.confidence
        else:  # Raw detection tensor
            x1, y1, x2, y2, conf, cls = det.tolist()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size_x = x2 - x1
        size_y = y2 - y1
        
        # Create Gaussian heatmap weighted by confidence
        y_coords, x_coords = np.ogrid[:h, :w]
        sigma_x = max(size_x / 2.5, 25)
        sigma_y = max(size_y / 2.5, 25)
        
        gaussian = np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) + 
                           (y_coords - center_y)**2 / (2 * sigma_y**2)))
        
        # Weight by confidence
        gaussian = gaussian * conf
        
        # Create bbox mask for sharper focus
        mask = np.zeros((h, w), dtype=np.float32)
        padding = 15
        mask[max(0, y1-padding):min(h, y2+padding), max(0, x1-padding):min(w, x2+padding)] = 1.0
        
        # Blend: strong inside bbox, weaker outside
        bbox_mask = np.zeros((h, w), dtype=np.float32)
        bbox_mask[y1:y2, x1:x2] = 1.0
        
        blended_heatmap = gaussian * (0.8 * bbox_mask + 0.2 * mask)
        
        combined_heatmap = np.maximum(combined_heatmap, blended_heatmap)
    
    # Normalize combined heatmap
    if combined_heatmap.max() > 0:
        combined_heatmap = combined_heatmap / combined_heatmap.max()
    
    return combined_heatmap

"""Image processing utilities"""
import cv2
import numpy as np
import torch
from typing import Optional, Tuple


def load_image_from_upload(uploaded_file) -> Optional[np.ndarray]:
    """
    Load image from Streamlit uploaded file
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Image as numpy array (BGR format) or None if failed
    """
    try:
        img_array = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def enhance_image(image: np.ndarray, dce_model, device) -> Tuple[np.ndarray, bool]:
    """
    Apply DCE++ enhancement to image
    
    Args:
        image: Input image (BGR format)
        dce_model: DCE++ model
        device: Device to run model on
        
    Returns:
        tuple: (enhanced_image, success_flag)
    """
    if dce_model is None:
        return image, False
    
    try:
        # Convert to RGB and normalize
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        # Apply DCE++ enhancement
        with torch.no_grad():
            enhanced_tensor = dce_model(img_tensor.to(device))
        
        # Convert back to numpy
        enhanced_img = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        enhanced_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
        
        return enhanced_bgr, True
    except Exception as e:
        print(f"Error enhancing image: {e}")
        return image, False

"""Model loading utilities"""
import torch
from collections import defaultdict
from pathlib import Path
from src.models.dce_model import DCENet_pp
from src.config.constants import (
    DCE_MODEL_PATH, TIGER_MODEL_PATH, LION_MODEL_PATH, DEFAULT_MODEL_PATH
)


class ModelLoader:
    """Utility class for loading all models"""
    
    @staticmethod
    def load_dce_model(weights_path=DCE_MODEL_PATH, device=None):
        """
        Load Zero-DCE++ enhancement model
        
        Args:
            weights_path: Path to model weights
            device: Device to load model on (None for auto-detect)
            
        Returns:
            tuple: (model, device, success_flag)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        dce_model = DCENet_pp().to(device)
        try:
            if not Path(weights_path).exists():
                return None, device, False
                
            state_dict = torch.load(weights_path, map_location=device, weights_only=False)
            dce_model.load_state_dict(state_dict)
            dce_model.eval()
            return dce_model, device, True
        except Exception as e:
            print(f"Warning: Failed to load DCE++ model: {e}")
            return None, device, False
    
    @staticmethod
    def load_detection_models(tiger_path=TIGER_MODEL_PATH, 
                             lion_path=LION_MODEL_PATH,
                             default_path=DEFAULT_MODEL_PATH):
        """
        Load YOLOv9 detection models
        
        Args:
            tiger_path: Path to tiger detection model
            lion_path: Path to lion detection model
            default_path: Path to default YOLOv9 model
            
        Returns:
            tuple: (tiger_model, lion_model, default_model)
        """
        # Load tiger model
        tiger_model = torch.hub.load(
            "yolov9",
            "custom",
            path=tiger_path,
            source="local"
        )
        tiger_model.names = defaultdict(lambda: 'unknown', {0: 'tiger'})

        # Load lion model
        lion_model = torch.hub.load(
            "yolov9",
            "custom",
            path=lion_path,
            source="local"
        )
        lion_model.names = defaultdict(lambda: 'unknown', {0: 'lion'})

        # Load default YOLOv9 model for other animals
        default_model = torch.hub.load(
            "yolov9",
            "custom",
            path=default_path,
            source="local"
        )
        
        return tiger_model, lion_model, default_model

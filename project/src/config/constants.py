"""Configuration constants for the application"""

# Wild animal classes from COCO dataset
WILD_ANIMALS = {
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
}

# Confidence thresholds
TIGER_CONFIDENCE_THRESHOLD = 0.8
LION_CONFIDENCE_THRESHOLD = 0.8
DEFAULT_CONFIDENCE_THRESHOLD = 0.8

# Model paths
DCE_MODEL_PATH = 'Epoch99.pth'
TIGER_MODEL_PATH = 'tiger.pt'
LION_MODEL_PATH = 'lion.pt'
DEFAULT_MODEL_PATH = 'yolov9-c.pt'

# Grad-CAM settings
GRADCAM_ALPHA = 0.5
GRADCAM_COLORMAP = 'jet'

# Supported image formats
SUPPORTED_IMAGE_FORMATS = ["jpg", "png", "webp", "jpeg"]

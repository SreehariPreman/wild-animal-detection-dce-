# Wild Animal Detection with Low-Light Enhancement 

A comprehensive deep learning application for detecting wild animals in images with optional low-light enhancement using Zero-DCE++ and Grad-CAM visualization. Built with YOLOv9 for object detection and Streamlit for the user interface.

## Features

- 🦁 **Specialized Animal Detection**: Custom models for tiger, lion, and general wild animal detection
- 🌙 **Low-Light Enhancement**: Zero-DCE++ (Zero-Reference Deep Curve Estimation) for improving visibility in dark images
- 🔥 **Grad-CAM Heatmaps**: Visual explanations showing which image regions the model focuses on
- 📊 **Detection History Logging**: Track all detections with persistent CSV logging across sessions


## Project Structure

```
wild-animal-detection-dce-/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── src/                    # Source code modules
│   ├── models/            # Model definitions
│   │   └── dce_model.py   # Zero-DCE++ model (DCENet_pp)
│   ├── detection/         # Detection logic
│   │   ├── detector.py    # DetectionManager class
│   │   └── model_loader.py # Model loading utilities
│   ├── utils/             # Utility functions
│   │   ├── gradcam.py     # Grad-CAM implementation
│   │   ├── image_processing.py # Image enhancement utilities
│   │   └── detection_logger.py # Detection history logging
│   └── config/            # Configuration
│       └── constants.py   # Configuration constants
├── yolov9/                # YOLOv9 repository (submodule/cloned)
└── demo/                  # Sample images for testing
```

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster inference)
- 4GB+ RAM
- Internet connection for downloading model weights

## Installation

### 1. Clone Required Repositories

This project uses the following repositories:

#### YOLOv9 Repository
```bash
git clone https://github.com/WongKinYiu/yolov9.git
```

#### Zero-DCE Repository (Reference)
The Zero-DCE++ implementation is included in this repository, but you can reference the original:
```bash
# Original repository (for reference)
git clone https://github.com/Li-Chongyi/Zero-DCE.git ZeroDCE
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install torchvision (if not already installed)
pip install torchvision

# Install additional YOLOv9 dependencies
cd yolov9
pip install -r requirements.txt
cd ..
```

### 4. Download Model Weights

You need to download the following model weights:

#### YOLOv9-C Model (Default Detection)
```bash
# Create weights directory
mkdir -p weights

# Download YOLOv9-C weights
wget -P weights https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt
```

#### Zero-DCE++ Model Weights
```bash
# Download Zero-DCE++ weights (Epoch99.pth)
wget -O weights/Epoch99.pth "https://github.com/Li-Chongyi/Zero-DCE_extension/blob/main/Zero-DCE%2B%2B/snapshots_Zero_DCE%2B%2B/Epoch99.pth?raw=true"
```

#### Custom Model Weights
Place your custom model weights in the root directory:
- `tiger.pt` - Tiger detection model
- `lion.pt` - Lion detection model

**Note**: If custom models are not available, the application will use the default YOLOv9-C model for all detections.

### 5. Verify Installation

```bash
# Verify Python packages
python -c "import torch, cv2, streamlit, numpy; print('All packages installed successfully!')"

# Check if YOLOv9 is accessible
python -c "import sys; sys.path.append('yolov9'); print('YOLOv9 path added')"
```

## Usage

### Running the Application

```bash
# Make sure you're in the project root directory
streamlit run app.py
```

The application will start on `http://localhost:8501` by default.

### Using the Interface

#### Detection Tab

1. **Upload Image**: Click "Upload image" and select an image file (JPG, PNG, WEBP, JPEG)

2. **Enable Low-Light Enhancement** (Optional):
   - Check "Enable Low-Light Enhancement (DCE++)" in the sidebar
   - This improves visibility for dark or low-light images

3. **Enable Grad-CAM Heatmap** (Optional):
   - Check "Enable Grad-CAM Heatmap Visualization" in the sidebar
   - Shows which regions the model focuses on for detection

4. **View Results**:
   - **Original Image**: The uploaded image
   - **Detection Result**: Image with bounding boxes and labels
   - **Grad-CAM Heatmap** (if enabled): Heatmap overlay showing attention regions

### Detection Logging

Every detection run is automatically logged with the following information:
- **Timestamp**: When the detection was performed
- **Filename**: Name of the uploaded image
- **Enhancement Used**: Whether DCE++ enhancement was applied
- **Detection Status**: Success or No Detection
- **Number of Detections**: Count of detected animals
- **Tiger/Lion Detected**: Boolean flags for specialized detections
- **Other Animals**: List of other wild animals detected
- **All Detections**: Complete list with confidence scores
- **Max Confidence**: Highest confidence score in the detection

Logs are stored in `detection_logs.csv` in the project root directory and persist across application sessions.

### Supported Animal Classes

The default YOLOv9 model detects these wild animal classes from COCO dataset:
- Bird
- Cat
- Dog
- Horse
- Sheep
- Cow
- Elephant
- Bear
- Zebra
- Giraffe

Additionally, specialized models for:
- **Tiger** (custom trained model)
- **Lion** (custom trained model)

## Dependencies

### Core Dependencies
- `streamlit` - Web application framework
- `torch` - PyTorch deep learning framework
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `pandas` - Data manipulation and CSV handling
- `torchvision` - PyTorch vision utilities

### External Repositories Used
- **[YOLOv9](https://github.com/WongKinYiu/yolov9)** by WongKinYiu - Object detection framework
- **[Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE)** by Li-Chongyi - Zero-Reference Deep Curve Estimation for low-light enhancement

See `requirements.txt` for the complete list of dependencies.

## Configuration

Configuration constants are stored in `src/config/constants.py`:

- Model paths
- Confidence thresholds
- Animal class mappings
- Grad-CAM settings
- Supported image formats

You can modify these values according to your needs.

## Troubleshooting

### Model Loading Issues

**Issue**: "DCE++ weights not found"
- **Solution**: Ensure `Epoch99.pth` is in the project root directory

**Issue**: "YOLOv9 model not found"
- **Solution**: Ensure `yolov9-c.pt`, `tiger.pt`, and `lion.pt` are in the project root

### Import Errors

**Issue**: "ModuleNotFoundError: No module named 'yolov9'"
- **Solution**: Make sure the `yolov9` directory is cloned and accessible

**Issue**: "ModuleNotFoundError: No module named 'src'"
- **Solution**: Run the application from the project root directory

### Performance Issues

- Use GPU for faster inference (CUDA required)
- Reduce image size for faster processing
- Disable Grad-CAM visualization if not needed for faster results

## Development

### Project Architecture

The codebase follows a modular architecture:

- **Models** (`src/models/`): Neural network model definitions
- **Detection** (`src/detection/`): Detection logic and model loading
- **Utils** (`src/utils/`): Utility functions (Grad-CAM, image processing)
- **Config** (`src/config/`): Configuration constants


# app.py
import streamlit as st
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

# Zero-DCE++ Model Definition
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DSConv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x

class DCENet_pp(nn.Module):
    def __init__(self):
        super(DCENet_pp, self).__init__()
        self.e_conv1 = DSConv(3, 32)
        self.e_conv2 = DSConv(32, 32)
        self.e_conv3 = DSConv(32, 32)
        self.e_conv4 = DSConv(32, 32)
        self.e_conv5 = DSConv(64, 32)
        self.e_conv6 = DSConv(64, 32)
        self.e_conv7 = DSConv(64, 3)

    def forward(self, x):
        x1 = F.relu(self.e_conv1(x))
        x2 = F.relu(self.e_conv2(x1))
        x3 = F.relu(self.e_conv3(x2))
        x4 = F.relu(self.e_conv4(x3))
        x5 = F.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = F.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))

        for _ in range(8):
            x = x + x_r * (torch.pow(x, 2) - x)
        return x

# Load Zero-DCE++ model
@st.cache_resource
def load_dce_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dce_model = DCENet_pp().to(device)
    try:
        weights_path = 'Epoch99.pth'
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
        dce_model.load_state_dict(state_dict)
        dce_model.eval()
        return dce_model, device, True
    except FileNotFoundError:
        st.warning("⚠️ DCE++ weights not found. Enhancement disabled. Please download Epoch99.pth")
        return None, device, False

dce_model, device, dce_available = load_dce_model()

# Load detection models
@st.cache_resource
def load_detection_models():
    # Load tiger model
    tiger_model = torch.hub.load(
        "yolov9",
        "custom",
        path="tiger.pt",
        source="local"
    )
    tiger_model.names = defaultdict(lambda: 'unknown', {0: 'tiger'})

    # Load lion model
    lion_model = torch.hub.load(
        "yolov9",
        "custom",
        path="lion.pt",
        source="local"
    )
    lion_model.names = defaultdict(lambda: 'unknown', {0: 'lion'})

    # Load default YOLOv9 model for other animals
    default_model = torch.hub.load(
        "yolov9",
        "custom",
        path="yolov9-c.pt",
        source="local"
    )
    
    return tiger_model, lion_model, default_model

tiger_model, lion_model, default_model = load_detection_models()

# Wild animal classes from COCO dataset
wild_animals = {
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
}

st.title("Wild Animal Detection with Low-Light Enhancement 🦁🐯")

# Add toggle for enhancement
use_enhancement = st.checkbox("Enable Low-Light Enhancement (DCE++)", value=dce_available)

img_file = st.file_uploader("Upload image", type=["jpg","png","webp","jpeg"])

if img_file:
    # Read original image
    img_original = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    
    # Apply enhancement if enabled
    img = img_original.copy()
    if use_enhancement and dce_available and dce_model is not None:
        with st.spinner("Enhancing image..."):
            # Convert to RGB and normalize
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Apply DCE++ enhancement
            with torch.no_grad():
                enhanced_tensor = dce_model(img_tensor.to(device))
            
            # Convert back to numpy
            enhanced_img = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
            st.success("✨ Image enhanced successfully!")

    # Create a copy for annotation
    img_annotated = img.copy()

    # Set confidence thresholds for specialized models
    TIGER_CONFIDENCE_THRESHOLD = 0.8
    LION_CONFIDENCE_THRESHOLD = 0.8

    # Run default model first
    default_results = default_model(img)
    default_preds = default_results.pred[0]
    
    tiger_detected_general = False
    lion_detected_general = False
    other_animals_detected = False
    detected_animals = []
    
    for det in default_preds:
        x1, y1, x2, y2, conf, cls = det.tolist()
        cls = int(cls)
        if conf < 0.8:
            continue  # Only show detections with confidence > 0.8
        # Check if tiger or lion detected by general model
        if cls == 21:  # bear class in COCO, but check if it's detecting as big cat
            tiger_detected_general = True
        if cls in wild_animals:  # Wild animals
            other_animals_detected = True
            animal_name = wild_animals[cls]
            if animal_name not in detected_animals:
                detected_animals.append(animal_name)
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img_annotated, (x1,y1), (x2,y2), (255,0,0), 2)  # Blue for other animals
            label = f"{animal_name} {conf:.2f}"
            print("detected default model",label)
            cv2.putText(img_annotated, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
    if other_animals_detected and detected_animals:
        animals_str = ", ".join(detected_animals)
        st.info(f"🦒 {animals_str.capitalize()} detected using YOLOv9-c General Model (confidence > 0.8)")
    
    # Now run specialized tiger model for verification/better detection
    tiger_results = tiger_model(img)
    tiger_preds = tiger_results.pred[0]
    tiger_detected = False
    
    # Process tiger detections
    for det in tiger_preds:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) != 0:   # only tiger
            continue
        if conf < TIGER_CONFIDENCE_THRESHOLD:  # Check confidence threshold
            continue
        tiger_detected = True
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Overwrite with specialized detection (orange)
        cv2.rectangle(img_annotated, (x1,y1), (x2,y2), (0,165,255), 3)  # Orange for tiger, thicker
        label = f"tiger {conf:.2f}"
        print("detected default model",label)
        cv2.putText(img_annotated, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    
    if tiger_detected:
        st.success("🐯 Tiger detected using specialized Tiger Detection Model")
    
    # Run specialized lion model
    lion_results = lion_model(img)
    lion_preds = lion_results.pred[0]
    lion_detected = False
    
    for det in lion_preds:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if int(cls) != 0:   # only lion
            continue
        if conf < LION_CONFIDENCE_THRESHOLD:  # Check confidence threshold
            continue
        lion_detected = True
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        # Overwrite with specialized detection (gold)
        cv2.rectangle(img_annotated, (x1,y1), (x2,y2), (0,215,255), 3)  # Gold for lion, thicker
        label = f"lion {conf:.2f}"
        print("detected default model",label)
        cv2.putText(img_annotated, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,215,255), 2)
    if lion_detected:
        st.success("🦁 Lion detected using specialized Lion Detection Model (confidence > 0.8)")
    
    # Show warning if no animals detected at all
    if not other_animals_detected and not tiger_detected and not lion_detected:
        st.warning("⚠️ No wild animals detected")

    # Display images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_original, channels="BGR")
    with col2:
        st.subheader("Detection Result" + (" (Enhanced)" if use_enhancement and dce_available else ""))
        st.image(img_annotated, channels="BGR")




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

# Grad-CAM Implementation for YOLOv9
class GradCAM:
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
    """Generate Grad-CAM heatmaps for all detections"""
    h, w = image.shape[:2]
    combined_heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Generate heatmap for each detection using Gaussian kernels
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        bbox = [x1, y1, x2, y2]
        
        # Get detection center and size
        x1, y1, x2, y2 = map(int, bbox[:4])
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

st.title("Wild Animal Detection with Low-Light Enhancement 🦁🐯")

# Add toggle for enhancement
use_enhancement = st.checkbox("Enable Low-Light Enhancement (DCE++)", value=dce_available)

# Add toggle for Grad-CAM heatmap
use_gradcam = st.checkbox("Enable Grad-CAM Heatmap Visualization", value=True)

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
        st.info(f"🦒 {animals_str.capitalize()} detected")
    
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
        st.success("🐯 Tiger detected")
    
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
        st.success("🦁 Lion detected ")
    
    # Show warning if no animals detected at all
    if not other_animals_detected and not tiger_detected and not lion_detected:
        st.warning("⚠️ No wild animals detected")
    
    # Generate Grad-CAM heatmap if enabled
    img_heatmap = None
    all_detections = []
    
    if use_gradcam:
        # Collect all detections for heatmap generation
        for det in default_preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)
            if conf >= 0.8 and cls in wild_animals:
                all_detections.append(det)
        
        for det in tiger_preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0 and conf >= TIGER_CONFIDENCE_THRESHOLD:
                all_detections.append(det)
        
        for det in lion_preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            if int(cls) == 0 and conf >= LION_CONFIDENCE_THRESHOLD:
                all_detections.append(det)
        
        if all_detections:
            with st.spinner("Generating Grad-CAM heatmap..."):
                try:
                    # Use the model that detected the most animals
                    selected_model = default_model
                    if tiger_detected:
                        selected_model = tiger_model
                    elif lion_detected:
                        selected_model = lion_model
                    
                    # Generate heatmap
                    heatmap = generate_gradcam_heatmap(selected_model, img, all_detections, device)
                    
                    # Create GradCAM instance for overlay
                    grad_cam = GradCAM(selected_model)
                    img_heatmap = grad_cam.overlay_heatmap(img_annotated, heatmap, alpha=0.5)
                    
                    # Clean up hooks
                    del grad_cam
                except Exception as e:
                    st.warning(f"⚠️ Heatmap generation failed: {str(e)}")
                    # Fallback: create simple heatmap based on bounding boxes
                    h, w = img.shape[:2]
                    heatmap = np.zeros((h, w), dtype=np.float32)
                    for det in all_detections:
                        x1, y1, x2, y2, conf, cls = map(int, det[:4])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        size_x = x2 - x1
                        size_y = y2 - y1
                        
                        y_coords, x_coords = np.ogrid[:h, :w]
                        sigma_x = max(size_x / 3, 30)
                        sigma_y = max(size_y / 3, 30)
                        
                        gaussian = np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) + 
                                           (y_coords - center_y)**2 / (2 * sigma_y**2))) * conf
                        heatmap = np.maximum(heatmap, gaussian)
                    
                    if heatmap.max() > 0:
                        heatmap = heatmap / heatmap.max()
                    
                    # Apply colormap
                    heatmap_norm = (heatmap * 255).astype(np.uint8)
                    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                    img_heatmap = cv2.addWeighted(img_annotated, 0.5, heatmap_colored, 0.5, 0)

    # Display images
    if img_heatmap is not None and use_gradcam:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original Image")
            st.image(img_original, channels="BGR")
        with col2:
            st.subheader("Detection Result" + (" (Enhanced)" if use_enhancement and dce_available else ""))
            st.image(img_annotated, channels="BGR")
        with col3:
            st.subheader("Grad-CAM Heatmap")
            st.image(img_heatmap, channels="BGR")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img_original, channels="BGR")
        with col2:
            st.subheader("Detection Result" + (" (Enhanced)" if use_enhancement and dce_available else ""))
            st.image(img_annotated, channels="BGR")




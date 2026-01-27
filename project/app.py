"""Main Streamlit application for Wild Animal Detection with Low-Light Enhancement"""
import streamlit as st
import cv2
import numpy as np

# Local imports
from src.detection.model_loader import ModelLoader
from src.detection.detector import DetectionManager
from src.utils.gradcam import GradCAM, generate_gradcam_heatmap
from src.utils.image_processing import enhance_image, load_image_from_upload
from src.config.constants import SUPPORTED_IMAGE_FORMATS, GRADCAM_ALPHA


# Page configuration
st.set_page_config(
    page_title="Wild Animal Detection",
    page_icon="🦁",
    layout="wide"
)


@st.cache_resource
def load_models():
    """Load all models (cached by Streamlit)"""
    # Load DCE++ model for image enhancement
    dce_model, device, dce_available = ModelLoader.load_dce_model()
    
    # Load detection models
    tiger_model, lion_model, default_model = ModelLoader.load_detection_models()
    
    return {
        'dce_model': dce_model,
        'dce_available': dce_available,
        'device': device,
        'tiger_model': tiger_model,
        'lion_model': lion_model,
        'default_model': default_model
    }


def main():
    """Main application function"""
    st.title("Wild Animal Detection with Low-Light Enhancement")
    
    # Load models
    models = load_models()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        use_enhancement = st.checkbox(
            "Enable Low-Light Enhancement (DCE++)", 
            value=models['dce_available']
        )
        use_gradcam = st.checkbox(
            "Enable Grad-CAM Heatmap Visualization", 
            value=True
        )
    
    # File uploader
    img_file = st.file_uploader(
        "Upload image", 
        type=SUPPORTED_IMAGE_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
    )
    
    if not img_file:
        st.info("👆 Please upload an image to get started")
        return
    
    # Load and process image
    img_original = load_image_from_upload(img_file)
    if img_original is None:
        st.error("❌ Failed to load image. Please try another image.")
        return
    
    # Apply enhancement if enabled
    img = img_original.copy()
    if use_enhancement and models['dce_available'] and models['dce_model'] is not None:
        with st.spinner("Enhancing image..."):
            img, enhanced = enhance_image(img_original, models['dce_model'], models['device'])
            if enhanced:
                st.success("✨ Image enhanced successfully!")
            else:
                st.warning("⚠️ Enhancement failed, using original image")
    
    # Initialize detection manager
    detection_manager = DetectionManager(
        tiger_model=models['tiger_model'],
        lion_model=models['lion_model'],
        default_model=models['default_model']
    )
    
    # Run detection
    with st.spinner("Detecting animals..."):
        detections, img_annotated = detection_manager.detect_all(img)
    
    # Get detection summary
    summary = detection_manager.get_detection_summary(detections)
    
    # Display detection results
    display_detection_results(summary)
    
    # Show warning if no animals detected at all
    if not any([summary['tiger_detected'], summary['lion_detected'], 
                summary['other_animals_detected']]):
        st.warning("⚠️ No wild animals detected")
    
    # Generate Grad-CAM heatmap if enabled
    img_heatmap = None
    if use_gradcam and detections:
        img_heatmap = generate_heatmap_visualization(
            detections, img, img_annotated, 
            models, detection_manager
        )
    
    # Display images
    display_images(img_original, img_annotated, img_heatmap, 
                   use_enhancement, models['dce_available'], use_gradcam)


def display_detection_results(summary: dict):
    """Display detection results in the UI"""
    if summary['tiger_detected']:
        st.success("🐯 Tiger detected")
    
    if summary['lion_detected']:
        st.success("🦁 Lion detected")
    
    if summary['other_animals_detected'] and summary['detected_animals']:
        animals_str = ", ".join(summary['detected_animals'])
        st.info(f"🦒 {animals_str.capitalize()} detected")


def generate_heatmap_visualization(detections, img, img_annotated, models, detection_manager):
    """Generate Grad-CAM heatmap visualization"""
    with st.spinner("Generating Grad-CAM heatmap..."):
        try:
            # Select model based on detections
            summary = detection_manager.get_detection_summary(detections)
            selected_model = models['default_model']
            if summary['tiger_detected']:
                selected_model = models['tiger_model']
            elif summary['lion_detected']:
                selected_model = models['lion_model']
            
            # Generate heatmap
            heatmap = generate_gradcam_heatmap(
                selected_model, img, detections, models['device']
            )
            
            # Create GradCAM instance for overlay
            grad_cam = GradCAM(selected_model)
            img_heatmap = grad_cam.overlay_heatmap(
                img_annotated, heatmap, alpha=GRADCAM_ALPHA
            )
            
            # Clean up hooks
            del grad_cam
            
            return img_heatmap
        except Exception as e:
            st.warning(f"⚠️ Heatmap generation failed: {str(e)}")
            # Fallback: create simple heatmap based on bounding boxes
            return create_fallback_heatmap(detections, img, img_annotated)


def create_fallback_heatmap(detections, img, img_annotated):
    """Create fallback heatmap when Grad-CAM fails"""
    h, w = img.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for det in detections:
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        size_x = x2 - x1
        size_y = y2 - y1
        
        y_coords, x_coords = np.ogrid[:h, :w]
        sigma_x = max(size_x / 3, 30)
        sigma_y = max(size_y / 3, 30)
        
        gaussian = np.exp(-((x_coords - center_x)**2 / (2 * sigma_x**2) + 
                           (y_coords - center_y)**2 / (2 * sigma_y**2))) * det.confidence
        heatmap = np.maximum(heatmap, gaussian)
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Apply colormap
    heatmap_norm = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    img_heatmap = cv2.addWeighted(img_annotated, 0.5, heatmap_colored, 0.5, 0)
    
    return img_heatmap


def display_images(img_original, img_annotated, img_heatmap, 
                   use_enhancement, dce_available, use_gradcam):
    """Display images in the UI"""
    if img_heatmap is not None and use_gradcam:
        # Three columns: Original, Detection, Heatmap
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Original Image")
            st.image(img_original, channels="BGR", use_container_width=True)
        with col2:
            st.subheader("Detection Result")
            st.image(img_annotated, channels="BGR", use_container_width=True)
        with col3:
            st.subheader("Grad-CAM Heatmap")
            st.image(img_heatmap, channels="BGR", use_container_width=True)
    else:
        # Two columns: Original, Detection
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(img_original, channels="BGR", use_container_width=True)
        with col2:
            st.subheader(f"Detection Result")
            st.image(img_annotated, channels="BGR", use_container_width=True)


if __name__ == "__main__":
    main()

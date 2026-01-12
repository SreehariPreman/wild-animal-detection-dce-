# app.py
import streamlit as st
import cv2
import torch
import numpy as np
from collections import defaultdict

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

# Wild animal classes from COCO dataset
wild_animals = {
    14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 
    19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
}

st.title("Wild Animal Detection 🦁🐯")

img_file = st.file_uploader("Upload image", type=["jpg","png"])

if img_file:
    img = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    # Run tiger model first
    tiger_results = tiger_model(img)
    tiger_preds = tiger_results.pred[0]
    tiger_detected = False
    
    # Process tiger detections
    for det in tiger_preds:
        x1, y1, x2, y2, _, cls = det.tolist()
        if int(cls) != 0:   # only tiger
            continue
        tiger_detected = True
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,165,255), 2)  # Orange for tiger
        cv2.putText(img, "tiger", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,165,255), 2)
    
    if tiger_detected:
        st.success("🐯 Tiger detected using Tiger Detection Model")
    
    # Only run lion model if no tigers detected
    lion_detected = False
    if not tiger_detected:
        lion_results = lion_model(img)
        lion_preds = lion_results.pred[0]
        
        for det in lion_preds:
            x1, y1, x2, y2, _, cls = det.tolist()
            if int(cls) != 0:   # only lion
                continue
            lion_detected = True
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,215,255), 2)  # Gold for lion
            cv2.putText(img, "lion", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,215,255), 2)
        
        if lion_detected:
            st.success("🦁 Lion detected using Lion Detection Model")
    
    # Run default model if neither tiger nor lion detected
    other_animals_detected = False
    if not tiger_detected and not lion_detected:
        default_results = default_model(img)
        default_preds = default_results.pred[0]
        
        detected_animals = []
        for det in default_preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)
            if cls in wild_animals:  # Only show wild animals
                other_animals_detected = True
                animal_name = wild_animals[cls]
                if animal_name not in detected_animals:
                    detected_animals.append(animal_name)
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 2)  # Blue for other animals
                cv2.putText(img, animal_name, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        
        if other_animals_detected:
            animals_str = ", ".join(detected_animals)
            st.info(f"🦒 {animals_str.capitalize()} detected using YOLOv9-c General Model")
        elif not tiger_detected and not lion_detected:
            st.warning("⚠️ No wild animals detected")

    st.image(img, channels="BGR")




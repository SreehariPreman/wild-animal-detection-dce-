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
    
    # Only run lion model if no tigers detected
    if not tiger_detected:
        lion_results = lion_model(img)
        lion_preds = lion_results.pred[0]
        
        for det in lion_preds:
            x1, y1, x2, y2, _, cls = det.tolist()
            if int(cls) != 0:   # only lion
                continue
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,215,255), 2)  # Gold for lion
            cv2.putText(img, "lion", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,215,255), 2)

    st.image(img, channels="BGR")



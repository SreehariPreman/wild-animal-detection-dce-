"""Detection logic and result processing"""
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from src.config.constants import (
    WILD_ANIMALS, TIGER_CONFIDENCE_THRESHOLD, 
    LION_CONFIDENCE_THRESHOLD, DEFAULT_CONFIDENCE_THRESHOLD
)


@dataclass
class DetectionResult:
    """Data class for detection results"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    animal_name: str
    model_type: str  # 'tiger', 'lion', or 'default'
    
    def to_bbox(self):
        """Return bounding box as list"""
        return [self.x1, self.y1, self.x2, self.y2]


class DetectionManager:
    """Manages detection across multiple models"""
    
    def __init__(self, tiger_model, lion_model, default_model):
        self.tiger_model = tiger_model
        self.lion_model = lion_model
        self.default_model = default_model
    
    def detect_all(self, image: np.ndarray) -> Tuple[List[DetectionResult], np.ndarray]:
        """
        Run all detection models and annotate image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            tuple: (list of detections, annotated image)
        """
        img_annotated = image.copy()
        all_detections = []
        
        # Run default model
        default_detections = self._detect_default(image)
        for det in default_detections:
            all_detections.append(det)
            self._draw_detection(img_annotated, det, color=(255, 0, 0), thickness=2)
        
        # Run tiger model
        tiger_detections = self._detect_tiger(image)
        for det in tiger_detections:
            all_detections.append(det)
            self._draw_detection(img_annotated, det, color=(0, 165, 255), thickness=3)
        
        # Run lion model
        lion_detections = self._detect_lion(image)
        for det in lion_detections:
            all_detections.append(det)
            self._draw_detection(img_annotated, det, color=(0, 215, 255), thickness=3)
        
        return all_detections, img_annotated
    
    def _detect_default(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect using default YOLOv9 model"""
        results = self.default_model(image)
        preds = results.pred[0]
        detections = []
        
        for det in preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            cls = int(cls)
            
            if conf < DEFAULT_CONFIDENCE_THRESHOLD:
                continue
            
            if cls in WILD_ANIMALS:
                animal_name = WILD_ANIMALS[cls]
                detections.append(DetectionResult(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=conf, class_id=cls, animal_name=animal_name,
                    model_type='default'
                ))
        
        return detections
    
    def _detect_tiger(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect tigers using specialized model"""
        results = self.tiger_model(image)
        preds = results.pred[0]
        detections = []
        
        for det in preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            if int(cls) != 0:  # only tiger class
                continue
            
            if conf < TIGER_CONFIDENCE_THRESHOLD:
                continue
            
            detections.append(DetectionResult(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=conf, class_id=0, animal_name='tiger',
                model_type='tiger'
            ))
        
        return detections
    
    def _detect_lion(self, image: np.ndarray) -> List[DetectionResult]:
        """Detect lions using specialized model"""
        results = self.lion_model(image)
        preds = results.pred[0]
        detections = []
        
        for det in preds:
            x1, y1, x2, y2, conf, cls = det.tolist()
            
            if int(cls) != 0:  # only lion class
                continue
            
            if conf < LION_CONFIDENCE_THRESHOLD:
                continue
            
            detections.append(DetectionResult(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                confidence=conf, class_id=0, animal_name='lion',
                model_type='lion'
            ))
        
        return detections
    
    @staticmethod
    def _draw_detection(image: np.ndarray, detection: DetectionResult, 
                       color: Tuple[int, int, int], thickness: int = 2):
        """Draw detection bounding box and label on image"""
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        label = f"{detection.animal_name} {detection.confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, thickness)
    
    def get_detection_summary(self, detections: List[DetectionResult]) -> dict:
        """Get summary of detections"""
        summary = {
            'tiger_detected': False,
            'lion_detected': False,
            'other_animals_detected': False,
            'detected_animals': []
        }
        
        for det in detections:
            if det.model_type == 'tiger':
                summary['tiger_detected'] = True
            elif det.model_type == 'lion':
                summary['lion_detected'] = True
            elif det.model_type == 'default':
                summary['other_animals_detected'] = True
                if det.animal_name not in summary['detected_animals']:
                    summary['detected_animals'].append(det.animal_name)
        
        return summary

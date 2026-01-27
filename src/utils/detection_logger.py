"""Detection logging functionality for tracking history"""
import csv
import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path

from src.detection.detector import DetectionResult


class DetectionLogger:
    """Manages detection logging to CSV file"""
    
    def __init__(self, log_file: str = "detection_logs.csv"):
        """
        Initialize logger
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = log_file
        self.fieldnames = [
            'timestamp',
            'filename',
            'enhancement_used',
            'detection_status',
            'num_detections',
            'tiger_detected',
            'lion_detected',
            'other_animals',
            'all_detections',
            'max_confidence'
        ]
        self._initialize_log_file()
    
    def _initialize_log_file(self):
        """Create log file with headers if it doesn't exist"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
    
    def log_detection(
        self,
        filename: str,
        detections: List[DetectionResult],
        enhancement_used: bool,
        summary: dict
    ):
        """
        Log a detection result
        
        Args:
            filename: Name of the processed image file
            detections: List of detection results
            enhancement_used: Whether DCE++ enhancement was used
            summary: Detection summary dictionary
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Determine detection status
        detection_status = "Success" if detections else "No Detection"
        
        # Get all detected animal names with confidence
        all_detections_str = ""
        max_confidence = 0.0
        
        if detections:
            detection_details = []
            for det in detections:
                detection_details.append(f"{det.animal_name}({det.confidence:.2f})")
                max_confidence = max(max_confidence, det.confidence)
            all_detections_str = ", ".join(detection_details)
        
        # Get other animals (excluding tiger and lion)
        other_animals = ", ".join(summary.get('detected_animals', []))
        
        # Create log entry
        log_entry = {
            'timestamp': timestamp,
            'filename': filename,
            'enhancement_used': 'Yes' if enhancement_used else 'No',
            'detection_status': detection_status,
            'num_detections': len(detections),
            'tiger_detected': 'Yes' if summary.get('tiger_detected', False) else 'No',
            'lion_detected': 'Yes' if summary.get('lion_detected', False) else 'No',
            'other_animals': other_animals,
            'all_detections': all_detections_str,
            'max_confidence': f"{max_confidence:.3f}" if detections else "N/A"
        }
        
        # Append to CSV
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(log_entry)
    
    def get_logs(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve detection logs
        
        Args:
            limit: Maximum number of recent logs to return (None for all)
            
        Returns:
            DataFrame containing logs
        """
        if not os.path.exists(self.log_file):
            return pd.DataFrame(columns=self.fieldnames)
        
        try:
            df = pd.read_csv(self.log_file)
            
            if limit is not None and len(df) > 0:
                df = df.tail(limit)
            
            return df
        except Exception as e:
            print(f"Error reading logs: {e}")
            return pd.DataFrame(columns=self.fieldnames)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from detection logs
        
        Returns:
            Dictionary containing statistics
        """
        df = self.get_logs()
        
        if df.empty:
            return {
                'total_detections': 0,
                'successful_detections': 0,
                'failed_detections': 0,
                'tiger_count': 0,
                'lion_count': 0,
                'enhancement_usage': 0
            }
        
        stats = {
            'total_detections': len(df),
            'successful_detections': len(df[df['detection_status'] == 'Success']),
            'failed_detections': len(df[df['detection_status'] == 'No Detection']),
            'tiger_count': len(df[df['tiger_detected'] == 'Yes']),
            'lion_count': len(df[df['lion_detected'] == 'Yes']),
            'enhancement_usage': len(df[df['enhancement_used'] == 'Yes'])
        }
        
        return stats
    
    def clear_logs(self):
        """Clear all logs (reset the CSV file)"""
        self._initialize_log_file()

"""
Evaluation Module
Computes accuracy, precision, recall, and F1-score for the detection system
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import os
import sys

# Handle imports for both direct execution and module import
try:
    from facial_landmarks import FacialLandmarkDetector
    from feature_extraction import FeatureExtractor
    from detector import DrowsinessDetector
except ImportError:
    # If running as script, add parent directory to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from src.facial_landmarks import FacialLandmarkDetector
    from src.feature_extraction import FeatureExtractor
    from src.detector import DrowsinessDetector


class Evaluator:
    """Evaluates the drowsiness detection system"""
    
    def __init__(self, ear_threshold=0.25, mar_threshold=0.5):
        """
        Initialize evaluator
        
        Args:
            ear_threshold: EAR threshold for drowsiness
            mar_threshold: MAR threshold for yawning
        """
        self.landmark_detector = FacialLandmarkDetector()
        self.feature_extractor = FeatureExtractor()
        self.detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold
        )
        
        self.results = []
    
    def evaluate_video(self, video_path, ground_truth=None):
        """
        Evaluate detection on a video file
        
        Args:
            video_path: Path to video file
            ground_truth: Optional ground truth labels (list of dicts with frame numbers)
            
        Returns:
            results: List of detection results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        frame_number = 0
        results = []
        
        print(f"Evaluating video: {video_path}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            landmarks, face_detected, _ = self.landmark_detector.detect_landmarks(frame)
            
            if face_detected:
                # Extract features
                left_eye, right_eye = self.landmark_detector.get_eye_landmarks(landmarks)
                mouth = self.landmark_detector.get_mouth_landmarks(landmarks)
                
                left_ear = self.feature_extractor.calculate_ear(left_eye)
                right_ear = self.feature_extractor.calculate_ear(right_eye)
                mar = self.feature_extractor.calculate_mar(mouth)
                head_pose = self.feature_extractor.calculate_head_pose(landmarks, frame.shape)
                
                # Update detector
                state = self.detector.update(left_ear, right_ear, mar, head_pose)
                
                # Get ground truth if available
                gt_drowsy = None
                gt_distracted = None
                if ground_truth:
                    for gt in ground_truth:
                        if gt['frame'] == frame_number:
                            gt_drowsy = gt.get('drowsy', False)
                            gt_distracted = gt.get('distracted', False)
                            break
                
                results.append({
                    'frame': frame_number,
                    'drowsy': state['drowsy'],
                    'distracted': state['distracted'],
                    'avg_ear': state['avg_ear'],
                    'mar': state['mar'],
                    'gt_drowsy': gt_drowsy,
                    'gt_distracted': gt_distracted
                })
            else:
                results.append({
                    'frame': frame_number,
                    'drowsy': False,
                    'distracted': False,
                    'avg_ear': None,
                    'mar': None,
                    'gt_drowsy': gt_drowsy if ground_truth else None,
                    'gt_distracted': gt_distracted if ground_truth else None
                })
            
            frame_number += 1
            
            if frame_number % 100 == 0:
                print(f"Processed {frame_number} frames...")
        
        cap.release()
        return results
    
    def compute_metrics(self, results):
        """
        Compute evaluation metrics
        
        Args:
            results: List of detection results with ground truth
            
        Returns:
            metrics: Dictionary of computed metrics
        """
        # Filter results with ground truth
        valid_results = [r for r in results if r['gt_drowsy'] is not None]
        
        if len(valid_results) == 0:
            print("Warning: No ground truth data available")
            return None
        
        # Extract predictions and ground truth
        y_true_drowsy = [1 if r['gt_drowsy'] else 0 for r in valid_results]
        y_pred_drowsy = [1 if r['drowsy'] else 0 for r in valid_results]
        
        y_true_distracted = [1 if r['gt_distracted'] else 0 for r in valid_results]
        y_pred_distracted = [1 if r['distracted'] else 0 for r in valid_results]
        
        # Compute metrics for drowsiness
        metrics = {
            'drowsiness': {
                'accuracy': accuracy_score(y_true_drowsy, y_pred_drowsy),
                'precision': precision_score(y_true_drowsy, y_pred_drowsy, zero_division=0),
                'recall': recall_score(y_true_drowsy, y_pred_drowsy, zero_division=0),
                'f1_score': f1_score(y_true_drowsy, y_pred_drowsy, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true_drowsy, y_pred_drowsy).tolist()
            },
            'distraction': {
                'accuracy': accuracy_score(y_true_distracted, y_pred_distracted),
                'precision': precision_score(y_true_distracted, y_pred_distracted, zero_division=0),
                'recall': recall_score(y_true_distracted, y_pred_distracted, zero_division=0),
                'f1_score': f1_score(y_true_distracted, y_pred_distracted, zero_division=0),
                'confusion_matrix': confusion_matrix(y_true_distracted, y_pred_distracted).tolist()
            }
        }
        
        return metrics
    
    def save_results(self, results, output_path):
        """
        Save evaluation results to file
        
        Args:
            results: List of detection results
            output_path: Path to save results
        """
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    def save_metrics(self, metrics, output_path):
        """
        Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            output_path: Path to save metrics
        """
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {output_path}")


def main():
    """Example evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate drowsiness detection system')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file')
    parser.add_argument('--ground-truth', type=str,
                       help='Path to ground truth JSON file')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR threshold')
    parser.add_argument('--mar-threshold', type=float, default=0.5,
                       help='MAR threshold')
    
    args = parser.parse_args()
    
    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth, 'r') as f:
            ground_truth = json.load(f)
    
    # Create evaluator
    evaluator = Evaluator(
        ear_threshold=args.ear_threshold,
        mar_threshold=args.mar_threshold
    )
    
    # Evaluate
    results = evaluator.evaluate_video(args.video, ground_truth)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save results
    results_path = os.path.join(args.output, 'results.csv')
    evaluator.save_results(results, results_path)
    
    # Compute and save metrics if ground truth available
    if ground_truth:
        metrics = evaluator.compute_metrics(results)
        if metrics:
            metrics_path = os.path.join(args.output, 'metrics.json')
            evaluator.save_metrics(metrics, metrics_path)
            
            # Print metrics
            print("\n" + "=" * 50)
            print("Evaluation Metrics:")
            print("=" * 50)
            print("\nDrowsiness Detection:")
            print(f"  Accuracy:  {metrics['drowsiness']['accuracy']:.4f}")
            print(f"  Precision: {metrics['drowsiness']['precision']:.4f}")
            print(f"  Recall:    {metrics['drowsiness']['recall']:.4f}")
            print(f"  F1-Score:  {metrics['drowsiness']['f1_score']:.4f}")
            print("\nDistraction Detection:")
            print(f"  Accuracy:  {metrics['distraction']['accuracy']:.4f}")
            print(f"  Precision: {metrics['distraction']['precision']:.4f}")
            print(f"  Recall:    {metrics['distraction']['recall']:.4f}")
            print(f"  F1-Score:  {metrics['distraction']['f1_score']:.4f}")
            print("=" * 50)


if __name__ == '__main__':
    main()


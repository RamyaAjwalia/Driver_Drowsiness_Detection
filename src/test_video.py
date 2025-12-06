"""
Test script for processing video files instead of live camera feed
"""

import cv2
import argparse
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.facial_landmarks import FacialLandmarkDetector
from src.feature_extraction import FeatureExtractor
from src.detector import DrowsinessDetector


def process_video(video_path, output_path=None, ear_threshold=0.25, mar_threshold=0.5):
    """
    Process a video file for drowsiness detection
    
    Args:
        video_path: Path to input video
        output_path: Path to save output video (optional)
        ear_threshold: EAR threshold
        mar_threshold: MAR threshold
    """
    # Initialize components
    landmark_detector = FacialLandmarkDetector()
    feature_extractor = FeatureExtractor()
    detector = DrowsinessDetector(
        ear_threshold=ear_threshold,
        mar_threshold=mar_threshold
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    
    # Initialize video writer if output path provided
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output will be saved to: {output_path}")
    
    frame_count = 0
    drowsy_count = 0
    distracted_count = 0
    
    print("\nProcessing video...")
    print("Press 'q' to quit, 'p' to pause")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect landmarks
                landmarks, face_detected, results = landmark_detector.detect_landmarks(frame)
                
                if face_detected:
                    # Draw landmarks
                    frame = landmark_detector.draw_landmarks(frame, results)
                    
                    # Extract features
                    left_eye, right_eye = landmark_detector.get_eye_landmarks(landmarks)
                    mouth = landmark_detector.get_mouth_landmarks(landmarks)
                    
                    left_ear = feature_extractor.calculate_ear(left_eye)
                    right_ear = feature_extractor.calculate_ear(right_eye)
                    mar = feature_extractor.calculate_mar(mouth)
                    head_pose = feature_extractor.calculate_head_pose(landmarks, frame.shape)
                    
                    # Update detector
                    state = detector.update(left_ear, right_ear, mar, head_pose)
                    
                    # Draw information
                    h, w = frame.shape[:2]
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
                    
                    # Display info
                    cv2.putText(frame, f'Frame: {frame_count}/{total_frames}', (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    if state['avg_ear'] is not None:
                        ear_color = (0, 255, 0) if state['avg_ear'] > detector.ear_threshold else (0, 0, 255)
                        cv2.putText(frame, f'EAR: {state["avg_ear"]:.3f}', (20, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ear_color, 2)
                    
                    if state['mar'] is not None:
                        mar_color = (0, 255, 0) if state['mar'] < detector.mar_threshold else (0, 0, 255)
                        cv2.putText(frame, f'MAR: {state["mar"]:.3f}', (20, 100),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mar_color, 2)
                    
                    # Status
                    if state['drowsy']:
                        cv2.putText(frame, 'ALERT: DROWSY!', (20, 160),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 10)
                        drowsy_count += 1
                    elif state['distracted']:
                        cv2.putText(frame, 'ALERT: DISTRACTED!', (20, 160),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 3)
                        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 165, 255), 10)
                        distracted_count += 1
                    else:
                        cv2.putText(frame, 'Status: ALERT', (20, 160),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Save frame if needed
                if video_writer:
                    video_writer.write(frame)
                
                frame_count += 1
                
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
            
            # Display frame
            cv2.imshow('Video Processing', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 50)
        print("Processing Complete:")
        print(f"Total frames: {frame_count}")
        print(f"Drowsy frames: {drowsy_count} ({100*drowsy_count/max(frame_count,1):.2f}%)")
        print(f"Distracted frames: {distracted_count} ({100*distracted_count/max(frame_count,1):.2f}%)")
        print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description='Process video file for drowsiness detection')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str,
                       help='Path to save output video (optional)')
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR threshold (default: 0.25)')
    parser.add_argument('--mar-threshold', type=float, default=0.5,
                       help='MAR threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    process_video(
        args.video,
        args.output,
        args.ear_threshold,
        args.mar_threshold
    )


if __name__ == '__main__':
    main()


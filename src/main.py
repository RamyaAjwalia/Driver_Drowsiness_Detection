"""
Main Application: Real-time Driver Drowsiness and Distraction Detection
"""

import cv2
import numpy as np
import argparse
from datetime import datetime
import os
import sys
import winsound  # For beep sounds on Windows
import threading
import time

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


class DriverMonitoringSystem:
    """Main driver monitoring system"""
    
    def __init__(self, 
                 camera_index=0,
                 ear_threshold=0.25,
                 mar_threshold=0.6,  # Increased to reduce false positives (higher = less sensitive)
                 head_pose_threshold=45.0,  # Increased to reduce false positives
                 disable_head_pose=False,
                 disable_beep=False,
                 save_output=False,
                 output_path='output'):
        """
        Initialize the monitoring system
        
        Args:
            camera_index: Camera device index
            ear_threshold: EAR threshold for drowsiness detection
            mar_threshold: MAR threshold for yawning detection
            head_pose_threshold: Head pose threshold in degrees
            disable_head_pose: Disable head pose detection
            save_output: Whether to save output video
            output_path: Path to save output files
        """
        self.camera_index = camera_index
        self.save_output = save_output
        self.output_path = output_path
        self.disable_head_pose = disable_head_pose
        
        # Initialize components
        self.landmark_detector = FacialLandmarkDetector()
        self.feature_extractor = FeatureExtractor()
        self.drowsiness_detector = DrowsinessDetector(
            ear_threshold=ear_threshold,
            mar_threshold=mar_threshold,
            head_pose_threshold=head_pose_threshold if not disable_head_pose else 999.0,  # Effectively disable if flag set
            consecutive_frames_head=150 if not disable_head_pose else 5,  # ~5 seconds for left/right head movement (150 frames at 30 FPS)
            consecutive_frames_ear=30,  # ~1 second for drowsiness (30 frames at 30 FPS)
            blink_max_frames=10  # Normal blink duration (~0.33 seconds, 10 frames at 30 FPS)
        )
        
        # Video writer
        self.video_writer = None
        
        # Statistics
        self.frame_count = 0
        self.drowsy_frames = 0
        self.distracted_frames = 0
        
        # Alert state tracking (to avoid spam)
        self._last_drowsy_alert = False
        self._last_distracted_alert = False
        
        # Beep settings
        self.enable_beep = not disable_beep  # Can be toggled
        self._last_beep_time = 0  # Track last beep time for continuous beeping
        self._beep_thread = None  # For continuous beeping
        self._stop_beeping = False  # Flag to stop beeping
        
        # For display
        self.last_left_ear = None
        self.last_right_ear = None
    
    def draw_info(self, image, state, fps):
        """
        Draw detection information on the image
        
        Args:
            image: Input image
            state: Detection state dictionary
            fps: Current FPS
            
        Returns:
            image: Image with information drawn
        """
        h, w = image.shape[:2]
        
        # Background for text
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # FPS
        cv2.putText(image, f'FPS: {fps:.1f}', (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # EAR - Show both left and right if available
        if state['avg_ear'] is not None:
            # Get individual EAR values if available
            left_ear_str = ""
            right_ear_str = ""
            if hasattr(self, 'last_left_ear') and self.last_left_ear is not None:
                left_ear_str = f"L:{self.last_left_ear:.3f} "
            if hasattr(self, 'last_right_ear') and self.last_right_ear is not None:
                right_ear_str = f"R:{self.last_right_ear:.3f} "
            
            # Show blink status
            blink_status = " [BLINK]" if state.get('is_blink', False) else ""
            ear_text = f'EAR: {left_ear_str}{right_ear_str}Avg:{state["avg_ear"]:.3f} (T:{self.drowsiness_detector.ear_threshold:.2f} C:{state["ear_counter"]}){blink_status}'
            
            # Color coding: green if open, yellow if blinking, red if closed too long
            if state.get('is_blink', False):
                color = (0, 255, 255)  # Yellow for blink
            elif state['avg_ear'] > self.drowsiness_detector.ear_threshold:
                color = (0, 255, 0)  # Green for open
            else:
                color = (0, 0, 255)  # Red for closed
            cv2.putText(image, ear_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        # MAR
        if state['mar'] is not None:
            mar_text = f'MAR: {state["mar"]:.3f}'
            color = (0, 255, 0) if state['mar'] < self.drowsiness_detector.mar_threshold else (0, 0, 255)
            cv2.putText(image, mar_text, (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Head pose
        if state['head_pose'] is not None:
            yaw = state["head_pose"].get("yaw", 0)
            pitch = state["head_pose"].get("pitch", 0)
            head_counter = state.get("head_counter", 0)
            head_normal_counter = getattr(self.drowsiness_detector, 'head_normal_counter', 0)
            
            # Show time remaining before alert (5 seconds = 150 frames)
            if head_counter > 0:
                time_remaining = max(0, (self.drowsiness_detector.consecutive_frames_head - head_counter) / 30.0)
                pose_text = f'Head: Y:{yaw:.1f} P:{pitch:.1f} C:{head_counter}/150 ({time_remaining:.1f}s)'
            else:
                pose_text = f'Head: Y:{yaw:.1f} P:{pitch:.1f} Normal ({head_normal_counter}/10)'
            
            # Color based on whether head is deviated and how long
            head_color = (0, 255, 0)  # Green if normal
            yaw_abs = abs(yaw)
            pitch_abs = abs(pitch)
            
            if state.get('distracted', False):
                head_color = (0, 0, 255)  # Red if alert active
            elif yaw_abs > self.drowsiness_detector.head_pose_threshold or pitch_abs > self.drowsiness_detector.head_pose_threshold:
                if head_counter >= self.drowsiness_detector.consecutive_frames_head:
                    head_color = (0, 0, 255)  # Red if alert triggered
                elif head_counter > self.drowsiness_detector.consecutive_frames_head * 0.7:  # >70% of threshold
                    head_color = (0, 100, 255)  # Orange-red if close to alert
                else:
                    head_color = (0, 165, 255)  # Orange if deviated but not yet alert
            cv2.putText(image, pose_text, (20, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, head_color, 2)
        
        # Status - Show both alerts if both conditions are true
        status_y = 160
        alert_active = False
        
        if state['drowsy']:
            # Drowsiness alert - Red
            cv2.putText(image, '!!! ALERT: DROWSY !!!', (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 4)
            # Draw red border (thicker for alert)
            cv2.rectangle(image, (0, 0), (w-1, h-1), (0, 0, 255), 15)
            alert_active = True
            status_y += 35  # Move next alert down
        
        if state['distracted']:
            # Distraction alert - Orange (only head pose now, yawning removed)
            distraction_text = '!!! ALERT: DISTRACTED !!!'
            
            cv2.putText(image, distraction_text, (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 4)
            # Draw orange border (thicker for alert)
            if not alert_active:  # Only draw border if drowsy alert didn't draw one
                cv2.rectangle(image, (0, 0), (w-1, h-1), (0, 165, 255), 15)
            else:
                # If both alerts, draw both borders (red outer, orange inner)
                cv2.rectangle(image, (5, 5), (w-6, h-6), (0, 165, 255), 10)
            alert_active = True
        
        if not alert_active:
            cv2.putText(image, 'Status: ALERT', (20, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Alert count and status
        alert_y = 200 if alert_active else 190
        alert_text = f'Total Alerts: {state["alert_count"]}'
        if state['drowsy']:
            alert_text += ' | DROWSY'
        if state['distracted']:
            alert_text += ' | DISTRACTED'
        cv2.putText(image, alert_text, (20, alert_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Beep status
        beep_status = "BEEP:ON" if self.enable_beep else "BEEP:OFF"
        beep_color = (0, 255, 0) if self.enable_beep else (128, 128, 128)
        cv2.putText(image, beep_status, (20, alert_y + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, beep_color, 1)
        
        return image
    
    def _beep_worker(self, frequency, duration, interval, stop_event):
        """Worker thread for continuous beeping"""
        while not stop_event.is_set():
            try:
                winsound.Beep(frequency, duration)
                time.sleep(interval)
            except Exception as e:
                print(f"[WARNING] Beep error: {e}")
                break
    
    def _handle_continuous_beeping(self, state):
        """Handle continuous beeping for active alerts with dynamic escalation"""
        if not self.enable_beep:
            # Stop any running beep thread
            if hasattr(self, '_beep_stop_event') and self._beep_stop_event:
                self._beep_stop_event.set()
            self._beep_thread = None
            return
        
        # Check if any alert is active (use actual state, not just flags)
        alert_active = state.get('drowsy', False) or state.get('distracted', False)
        
        if alert_active:
            # Determine beep parameters based on alert type and duration
            alert_duration = state.get('alert_duration_seconds', 0)
            
            if state.get('drowsy', False):
                # Drowsiness: high-pitched beeps
                if alert_duration > 5.0:
                    # After 5 seconds: 3x faster and higher frequency (louder perception)
                    frequency = 1200  # Higher frequency (perceived as louder)
                    duration = 100  # Faster (100ms instead of 200ms)
                    interval = 0.1  # 3x faster (0.1s instead of 0.3s)
                else:
                    frequency = 1000
                    duration = 200
                    interval = 0.3
            else:
                # Distraction: medium-pitched beeps
                if alert_duration > 5.0:
                    # After 5 seconds: 3x faster and higher frequency
                    frequency = 1000  # Higher frequency (perceived as louder)
                    duration = 150  # Faster (150ms instead of 300ms)
                    interval = 0.15  # 3x faster (0.15s instead of 0.45s)
                else:
                    frequency = 800
                    duration = 300
                    interval = 0.45
            
            # Start beeping thread if not already running
            if self._beep_thread is None or not self._beep_thread.is_alive():
                # Stop any existing beep thread first
                if hasattr(self, '_beep_stop_event') and self._beep_stop_event:
                    self._beep_stop_event.set()
                
                self._beep_stop_event = threading.Event()
                self._beep_thread = threading.Thread(
                    target=self._beep_worker,
                    args=(frequency, duration, interval, self._beep_stop_event),
                    daemon=True
                )
                self._beep_thread.start()
        else:
            # Stop beeping immediately if no alert
            if hasattr(self, '_beep_stop_event') and self._beep_stop_event:
                self._beep_stop_event.set()
            # Wait a bit for thread to stop, then clear reference
            if self._beep_thread is not None and self._beep_thread.is_alive():
                time.sleep(0.1)  # Brief wait for thread to stop
            self._beep_thread = None
    
    def process_frame(self, frame):
        """
        Process a single frame
        
        Args:
            frame: Input frame
            
        Returns:
            processed_frame: Frame with annotations
            state: Detection state
        """
        # Detect landmarks
        landmarks, face_detected, results = self.landmark_detector.detect_landmarks(frame)
        
        if not face_detected:
            return frame, {
                'drowsy': False,
                'distracted': False,
                'yawning': False,
                'head_distracted': False,
                'avg_ear': None,
                'mar': None,
                'head_pose': None,
                'ear_counter': 0,
                'mar_counter': 0,
                'alert_count': self.drowsiness_detector.alert_count
            }
        
        # Draw landmarks
        frame = self.landmark_detector.draw_landmarks(frame, results)
        
        # Extract features
        left_eye, right_eye = self.landmark_detector.get_eye_landmarks(landmarks)
        mouth = self.landmark_detector.get_mouth_landmarks(landmarks)
        
        # Calculate EAR
        left_ear = self.feature_extractor.calculate_ear(left_eye)
        right_ear = self.feature_extractor.calculate_ear(right_eye)
        
        # Store for display
        self.last_left_ear = left_ear
        self.last_right_ear = right_ear
        
        # Calculate MAR
        mar = self.feature_extractor.calculate_mar(mouth)
        
        # Calculate head pose (skip if disabled)
        head_pose = None
        if not self.disable_head_pose:
            head_pose = self.feature_extractor.calculate_head_pose(landmarks, frame.shape)
        
        # Update detector
        state = self.drowsiness_detector.update(left_ear, right_ear, mar, head_pose)
        
        # Update statistics - only count if not a normal blink
        # Drowsy: only count if eyes closed for more than 1 second (not a blink)
        if state['drowsy']:
            self.drowsy_frames += 1
        # Distracted: only count if sustained distraction (not brief mirror check)
        if state['distracted'] and not state.get('is_blink', False):
            self.distracted_frames += 1
        
        return frame, state
    
    def run(self):
        """Run the main detection loop"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {self.camera_index}")
            return
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        print(f"Camera initialized: {width}x{height} @ {fps} FPS")
        
        # Initialize video writer if needed
        if self.save_output:
            os.makedirs(self.output_path, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_path, f"output_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
            print(f"Saving output to: {output_file}")
        
        # FPS calculation
        fps_counter = 0
        fps_start_time = cv2.getTickCount()
        current_fps = 0
        
        print("\nDriver Monitoring System Started")
        print("Press 'q' to quit, 'r' to reset statistics")
        print("-" * 50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Process frame
                processed_frame, state = self.process_frame(frame)
                
                # Calculate FPS
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps_end_time = cv2.getTickCount()
                    time_diff = (fps_end_time - fps_start_time) / cv2.getTickFrequency()
                    current_fps = 30 / time_diff
                    fps_start_time = fps_end_time
                
                # Draw information
                processed_frame = self.draw_info(processed_frame, state, current_fps)
                
                # Save frame if needed
                if self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # Display frame
                cv2.imshow('Driver Drowsiness Detection', processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.drowsiness_detector.reset()
                    self.drowsy_frames = 0
                    self.distracted_frames = 0
                    self.frame_count = 0
                    print("Statistics reset")
                elif key == ord('b'):
                    # Toggle beep sound
                    self.enable_beep = not self.enable_beep
                    status = "enabled" if self.enable_beep else "disabled"
                    print(f"Beep sound {status}")
                
                self.frame_count += 1
                
                # Print alerts with more detail (only when state changes)
                if state['drowsy'] and not self._last_drowsy_alert:
                    print(f"[ALERT] ⚠️  DROWSINESS DETECTED at frame {self.frame_count} - Eyes closed for >1 second!")
                    self._last_drowsy_alert = True
                elif not state['drowsy'] and self._last_drowsy_alert:
                    print(f"[INFO] Drowsiness cleared at frame {self.frame_count}")
                    self._last_drowsy_alert = False
                
                if state['distracted'] and not self._last_distracted_alert:
                    print(f"[ALERT] ⚠️  DISTRACTION DETECTED at frame {self.frame_count} - Head turned away from road for >5 seconds!")
                    self._last_distracted_alert = True
                elif not state['distracted'] and self._last_distracted_alert:
                    print(f"[INFO] Distraction cleared at frame {self.frame_count}")
                    self._last_distracted_alert = False
                
                # Handle continuous beeping for active alerts
                self._handle_continuous_beeping(state)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Stop beeping
            if hasattr(self, '_beep_stop_event') and self._beep_stop_event:
                self._beep_stop_event.set()
            if self._beep_thread is not None:
                self._beep_thread.join(timeout=1.0)
            
            # Cleanup
            cap.release()
            if self.video_writer is not None:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            # Print statistics
            print("\n" + "=" * 50)
            print("Session Statistics:")
            print(f"Total frames: {self.frame_count}")
            print(f"Drowsy frames: {self.drowsy_frames} ({100*self.drowsy_frames/max(self.frame_count,1):.2f}%)")
            print(f"Distracted frames: {self.distracted_frames} ({100*self.distracted_frames/max(self.frame_count,1):.2f}%)")
            print(f"Total alerts: {self.drowsiness_detector.alert_count}")
            print("=" * 50)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Driver Drowsiness and Distraction Detection System'
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device index (default: 0)')
    parser.add_argument('--ear-threshold', type=float, default=0.25,
                       help='EAR threshold for drowsiness (default: 0.25, lower = more sensitive)')
    parser.add_argument('--mar-threshold', type=float, default=0.5,
                       help='MAR threshold for yawning (default: 0.5, higher = more sensitive)')
    parser.add_argument('--head-pose-threshold', type=float, default=40.0,
                       help='Head pose threshold in degrees (default: 40.0, higher = less sensitive)')
    parser.add_argument('--disable-head-pose', action='store_true',
                       help='Disable head pose detection (useful for testing)')
    parser.add_argument('--disable-beep', action='store_true',
                       help='Disable beep sound alerts')
    parser.add_argument('--save-output', action='store_true',
                       help='Save output video to file')
    parser.add_argument('--output-path', type=str, default='output',
                       help='Path to save output files (default: output)')
    
    args = parser.parse_args()
    
    # Create and run system
    system = DriverMonitoringSystem(
        camera_index=args.camera,
        ear_threshold=args.ear_threshold,
        mar_threshold=args.mar_threshold,
        head_pose_threshold=args.head_pose_threshold,
        disable_head_pose=args.disable_head_pose,
        disable_beep=args.disable_beep,
        save_output=args.save_output,
        output_path=args.output_path
    )
    
    system.run()


if __name__ == '__main__':
    main()


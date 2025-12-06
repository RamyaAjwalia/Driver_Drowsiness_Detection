"""
Drowsiness and Distraction Detection Module
Implements classification logic based on extracted features
"""

import numpy as np
from collections import deque


class DrowsinessDetector:
    """Detects driver drowsiness and distraction based on facial features"""
    
    def __init__(self, 
                 ear_threshold=0.25,
                 mar_threshold=0.5,
                 consecutive_frames_ear=30,  # ~1 second at 30 FPS
                 consecutive_frames_mar=15,  # ~0.5 seconds at 30 FPS (increased to reduce false positives)
                 head_pose_threshold=30.0,
                 consecutive_frames_head=150,  # ~5 seconds at 30 FPS for left/right head movement
                 blink_max_frames=10,  # Maximum frames for a normal blink (~0.33 seconds)
                 ear_history_size=10):
        """
        Initialize the drowsiness detector
        
        Args:
            ear_threshold: Threshold for eye closure detection (lower = more sensitive)
            mar_threshold: Threshold for yawning detection (higher = more sensitive)
            consecutive_frames_ear: Frames of closed eyes to trigger drowsiness (~1 second)
            consecutive_frames_mar: Frames of yawning to trigger distraction
            head_pose_threshold: Head angle threshold in degrees for distraction
            consecutive_frames_head: Frames of head deviation to trigger distraction (~5 seconds for left/right head movement)
            blink_max_frames: Maximum frames for a normal blink (ignore these)
            ear_history_size: Size of EAR history buffer for averaging
        """
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.consecutive_frames_ear = consecutive_frames_ear
        self.consecutive_frames_mar = consecutive_frames_mar
        self.head_pose_threshold = head_pose_threshold
        self.consecutive_frames_head = consecutive_frames_head
        self.blink_max_frames = blink_max_frames
        
        # Counters for consecutive frames
        self.ear_counter = 0
        self.mar_counter = 0
        self.head_counter = 0
        self.head_normal_counter = 0  # Counter for normal head position
        
        # Blink detection (to ignore normal blinks)
        self.blink_counter = 0
        self.in_blink = False
        
        # History buffers
        self.ear_history = deque(maxlen=ear_history_size)
        self.mar_history = deque(maxlen=ear_history_size)
        
        # State tracking
        self.is_drowsy = False
        self.is_distracted = False
        self.alert_count = 0
        self.last_drowsy_state = False
        self.last_distracted_state = False
    
    def update(self, left_ear, right_ear, mar, head_pose):
        """
        Update detector state with new features
        
        Args:
            left_ear: Left eye aspect ratio
            right_ear: Right eye aspect ratio
            mar: Mouth aspect ratio
            head_pose: Dictionary with 'pitch', 'yaw', 'roll' angles
            
        Returns:
            state: Dictionary with detection results
        """
        # Calculate average EAR
        avg_ear = None
        if left_ear is not None and right_ear is not None:
            avg_ear = (left_ear + right_ear) / 2.0
            self.ear_history.append(avg_ear)
        
        # Update MAR history
        if mar is not None:
            self.mar_history.append(mar)
        
        # Check for drowsiness (eye closure) - ignore normal blinks
        drowsy = False
        if avg_ear is not None:
            if avg_ear < self.ear_threshold:
                self.ear_counter += 1
                self.blink_counter += 1
                self.in_blink = True
            else:
                # Eyes opened - check if it was just a blink
                if self.in_blink:
                    # If closure was short (normal blink), reset counters
                    if self.blink_counter <= self.blink_max_frames:
                        # This was a normal blink, reset everything
                        self.ear_counter = 0
                        self.blink_counter = 0
                        self.in_blink = False
                    else:
                        # This was a longer closure, keep the counter but mark blink as ended
                        self.blink_counter = 0
                        self.in_blink = False
                else:
                    # Eyes were already open, reset counter
                    self.ear_counter = 0
            
            # Only trigger drowsiness if eyes closed for more than blink duration
            # AND more than the consecutive frames threshold
            if (self.ear_counter >= self.consecutive_frames_ear and 
                self.ear_counter > self.blink_max_frames):
                drowsy = True
                # Only increment alert count when transitioning from not drowsy to drowsy
                if not self.last_drowsy_state:
                    self.alert_count += 1
        
        # Yawning detection removed - only using head pose for distraction
        
        # Check for head pose distraction
        # In driving environment, left/right head movement is normal (mirror checking)
        # Only count as distracted if head is turned for more than 5 seconds
        head_distracted = False
        if head_pose is not None:
            # Get yaw angle with sign (negative = left, positive = right)
            yaw_angle_raw = head_pose.get('yaw', 0)
            yaw_angle = abs(yaw_angle_raw)  # Absolute value for threshold checking
            pitch_angle = abs(head_pose.get('pitch', 0))
            
            # Validate that angles are reasonable (not NaN or extremely large)
            # Use stricter validation to avoid false positives
            if (not np.isnan(yaw_angle) and not np.isnan(pitch_angle) and 
                yaw_angle < 80 and pitch_angle < 80 and
                yaw_angle >= 0 and pitch_angle >= 0):
                
                # Check for head deviation (left/right or up/down)
                # For driving: left/right (yaw) is normal for mirrors, but up/down (pitch) is more concerning
                head_deviated = False
                
                # Use stricter thresholds to reduce false positives
                # Only count pitch (up/down) as immediate concern, or extreme yaw
                if pitch_angle > self.head_pose_threshold:
                    # Looking up/down is always concerning
                    head_deviated = True
                elif yaw_angle > self.head_pose_threshold * 1.5:  # Extreme left/right (45+ degrees)
                    # Extreme left/right is concerning even if brief
                    head_deviated = True
                elif yaw_angle > self.head_pose_threshold:
                    # Moderate left/right - detect both left AND right directions
                    # Positive yaw = right, Negative yaw = left
                    head_deviated = True
                
                if head_deviated:
                    self.head_counter += 1
                    self.head_normal_counter = 0  # Reset normal counter when deviated
                else:
                    # Head is in normal position
                    self.head_normal_counter += 1
                    # Only reset head_counter if head has been normal for a few frames (debounce)
                    # This prevents false clears when head briefly returns to center
                    if self.head_normal_counter >= 10:  # ~0.33 seconds of normal position
                        self.head_counter = 0
                
                # Only trigger if head deviation persists for consecutive frames (5+ seconds)
                # This allows normal mirror checking without false alarms
                # Driver can look left/right for up to 5 seconds before alert
                if self.head_counter >= self.consecutive_frames_head:
                    head_distracted = True
                else:
                    # Clear distraction if head has been normal for enough frames
                    if self.head_normal_counter >= 10:
                        head_distracted = False
            else:
                # Reset counters if angles are invalid
                self.head_counter = 0
                self.head_normal_counter = 0
        else:
            # No head pose data - reset counters
            self.head_counter = 0
            self.head_normal_counter = 0
        
        # Overall distraction state (only head pose, yawning removed)
        distracted = head_distracted
        
        # Increment alert count for distraction (only when transitioning from not distracted to distracted)
        if distracted and not self.last_distracted_state:
            self.alert_count += 1
        
        # Update state
        self.is_drowsy = drowsy
        self.is_distracted = distracted
        self.last_drowsy_state = drowsy
        self.last_distracted_state = distracted
        
        # Determine if current state is a normal blink (should not be counted)
        is_blink = (self.in_blink and self.blink_counter <= self.blink_max_frames) or \
                   (self.ear_counter > 0 and self.ear_counter <= self.blink_max_frames)
        
        return {
            'drowsy': drowsy,
            'distracted': distracted,
            'yawning': False,  # Yawning detection removed
            'head_distracted': head_distracted,
            'avg_ear': avg_ear,
            'mar': mar,
            'head_pose': head_pose,
            'ear_counter': self.ear_counter,
            'mar_counter': 0,  # Not used anymore
            'head_counter': self.head_counter,
            'blink_counter': self.blink_counter,
            'is_blink': is_blink,
            'alert_duration_seconds': max(0, (self.head_counter - self.consecutive_frames_head) / 30.0) if head_distracted else 0,
            'alert_count': self.alert_count
        }
    
    def reset(self):
        """Reset detector state"""
        self.ear_counter = 0
        self.mar_counter = 0
        self.head_counter = 0
        self.head_normal_counter = 0
        self.blink_counter = 0
        self.in_blink = False
        self.is_drowsy = False
        self.is_distracted = False
        self.last_drowsy_state = False
        self.last_distracted_state = False
        self.ear_history.clear()
        self.mar_history.clear()
        self.alert_count = 0
    
    def get_ear_average(self):
        """Get average EAR from history"""
        if len(self.ear_history) == 0:
            return None
        return np.mean(self.ear_history)
    
    def get_mar_average(self):
        """Get average MAR from history"""
        if len(self.mar_history) == 0:
            return None
        return np.mean(self.mar_history)


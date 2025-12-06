"""
Configuration file for Driver Drowsiness Detection System
Modify these parameters to tune the detection sensitivity
"""

# Detection Thresholds
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold (lower = more sensitive to drowsiness)
MAR_THRESHOLD = 0.5   # Mouth Aspect Ratio threshold (higher = more sensitive to yawning)

# Consecutive Frame Counts
CONSECUTIVE_FRAMES_EAR = 3  # Frames of closed eyes to trigger drowsiness alert
CONSECUTIVE_FRAMES_MAR = 5  # Frames of yawning to trigger distraction alert

# Head Pose Thresholds (in degrees)
HEAD_POSE_THRESHOLD = 30.0  # Head angle threshold for distraction detection

# Camera Settings
CAMERA_INDEX = 0  # Default camera device index

# MediaPipe Settings
MIN_DETECTION_CONFIDENCE = 0.5  # Minimum confidence for face detection
MIN_TRACKING_CONFIDENCE = 0.5   # Minimum confidence for landmark tracking

# Display Settings
SHOW_LANDMARKS = True  # Show facial landmarks on video
SHOW_FPS = True        # Show FPS counter

# Output Settings
SAVE_OUTPUT = False           # Save output video
OUTPUT_DIRECTORY = "output"   # Directory for output files


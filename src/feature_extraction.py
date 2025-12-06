"""
Feature Extraction Module
Computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and Head Pose
"""

import numpy as np
import cv2


class FeatureExtractor:
    """Extracts features for drowsiness and distraction detection"""
    
    def __init__(self):
        """Initialize the feature extractor"""
        # Camera calibration parameters (approximate, should be calibrated for production)
        self.camera_matrix = np.array([
            [800, 0, 320],
            [0, 800, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self.dist_coeffs = np.zeros((4, 1))
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR)
        
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        where p1-p6 are vertical and horizontal eye landmarks
        
        Args:
            eye_landmarks: Array of eye landmark coordinates (16 points from MediaPipe)
            
        Returns:
            ear: Eye Aspect Ratio value
        """
        if eye_landmarks is None or len(eye_landmarks) < 16:
            return None
        
        # MediaPipe eye landmarks order for left eye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # MediaPipe eye landmarks order for right eye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # We need: outer corner, inner corner, top, bottom points
        
        # For the 16-point eye array from MediaPipe:
        # Index 0: outer corner (33 for left, 362 for right)
        # Index 8: inner corner (133 for left, 263 for right) 
        # Index 4: bottom point (145 for left, 374 for right)
        # Index 12: top point (159 for left, 386 for right)
        # Also use points around these for better accuracy
        
        try:
            # Horizontal points (outer and inner corners)
            p1 = eye_landmarks[0]   # Outer corner
            p4 = eye_landmarks[8]   # Inner corner
            
            # Vertical points (top and bottom)
            p2 = eye_landmarks[12]  # Top point
            p3 = eye_landmarks[11]  # Top-middle point
            p5 = eye_landmarks[5]   # Bottom-middle point  
            p6 = eye_landmarks[4]   # Bottom point
            
            # Calculate distances
            vertical_dist_1 = np.linalg.norm(p2 - p6)
            vertical_dist_2 = np.linalg.norm(p3 - p5)
            horizontal_dist = np.linalg.norm(p1 - p4)
            
            # Avoid division by zero
            if horizontal_dist == 0:
                return None
            
            # Calculate EAR
            ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
            
            return ear
        except (IndexError, ValueError):
            # Fallback: try with any available points
            if len(eye_landmarks) >= 6:
                try:
                    p1 = eye_landmarks[0]
                    p4 = eye_landmarks[-1] if len(eye_landmarks) > 8 else eye_landmarks[8]
                    p2 = eye_landmarks[len(eye_landmarks)//4]
                    p6 = eye_landmarks[len(eye_landmarks)*3//4]
                    
                    vertical_dist = np.linalg.norm(p2 - p6)
                    horizontal_dist = np.linalg.norm(p1 - p4)
                    
                    if horizontal_dist == 0:
                        return None
                    
                    ear = vertical_dist / horizontal_dist
                    return ear
                except:
                    return None
            return None
    
    def calculate_mar(self, mouth_landmarks):
        """
        Calculate Mouth Aspect Ratio (MAR)
        
        MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)
        
        Args:
            mouth_landmarks: Array of mouth landmark coordinates
            
        Returns:
            mar: Mouth Aspect Ratio value
        """
        if mouth_landmarks is None or len(mouth_landmarks) < 8:
            return None
        
        # Use key mouth points
        # Outer mouth corners
        p1 = mouth_landmarks[0]   # Left corner
        p5 = mouth_landmarks[6]   # Right corner
        
        # Vertical mouth points
        p2 = mouth_landmarks[2]   # Top
        p3 = mouth_landmarks[3]   # Top-middle
        p4 = mouth_landmarks[4]   # Middle
        p6 = mouth_landmarks[7]   # Bottom-middle
        p7 = mouth_landmarks[8]   # Bottom
        p8 = mouth_landmarks[9]   # Bottom
        
        # Calculate distances
        vertical_dist_1 = np.linalg.norm(p2 - p8)
        vertical_dist_2 = np.linalg.norm(p3 - p7)
        vertical_dist_3 = np.linalg.norm(p4 - p6)
        horizontal_dist = np.linalg.norm(p1 - p5)
        
        # Avoid division by zero
        if horizontal_dist == 0:
            return None
        
        # Calculate MAR
        mar = (vertical_dist_1 + vertical_dist_2 + vertical_dist_3) / (3.0 * horizontal_dist)
        
        return mar
    
    def calculate_head_pose(self, landmarks, image_shape):
        """
        Calculate head pose (pitch, yaw, roll) using solvePnP
        
        Args:
            landmarks: Full facial landmarks array
            image_shape: Shape of the image (height, width)
            
        Returns:
            angles: Dictionary with 'pitch', 'yaw', 'roll' in degrees, or None if calculation fails
        """
        if landmarks is None or len(landmarks) < 468:
            return None
        
        try:
            # Indices for 2D facial points (MediaPipe Face Mesh)
            # These correspond to the 3D model points
            face_2d_indices = [1, 33, 61, 199, 291, 405]
            
            # Extract 2D image points (only x, y coordinates, not z)
            # solvePnP requires points in shape (n, 2)
            image_points = np.array([
                [landmarks[1][0], landmarks[1][1]],    # Nose tip
                [landmarks[33][0], landmarks[33][1]],  # Chin
                [landmarks[61][0], landmarks[61][1]],  # Left eye left corner
                [landmarks[199][0], landmarks[199][1]], # Right eye right corner
                [landmarks[291][0], landmarks[291][1]], # Left mouth corner
                [landmarks[405][0], landmarks[405][1]]  # Right mouth corner
            ], dtype=np.float64)
            
            # Validate that we have at least 4 points
            if len(image_points) < 4:
                return None
            
            # Scale model points to match image scale
            h, w = image_shape[:2]
            scale = min(w, h) / 500.0
            model_points = self.model_points * scale
            
            # Ensure model_points and image_points have matching number of points
            if len(model_points) != len(image_points):
                # Use only the first 6 points from model_points
                model_points = model_points[:len(image_points)]
            
            # Solve PnP to get rotation and translation vectors
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None
            
            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Extract Euler angles
            angles = self._rotation_matrix_to_euler_angles(rotation_matrix)
            
            return {
                'pitch': angles[0],  # Nodding up/down
                'yaw': angles[1],    # Turning left/right
                'roll': angles[2]    # Tilting left/right
            }
        except (IndexError, ValueError, cv2.error) as e:
            # Return None if there's any error in calculation
            return None
    
    def _rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (pitch, yaw, roll)
        
        Args:
            R: 3x3 rotation matrix
            
        Returns:
            angles: Tuple of (pitch, yaw, roll) in degrees
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)
        
        return (pitch, yaw, roll)


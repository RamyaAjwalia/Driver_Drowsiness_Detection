"""
Facial Landmark Detection Module
Uses MediaPipe for real-time facial landmark detection
"""

import cv2
import mediapipe as mp
import numpy as np


class FacialLandmarkDetector:
    """Detects facial landmarks using MediaPipe Face Mesh"""
    
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the facial landmark detector
        
        Args:
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # MediaPipe Face Mesh has 468 landmarks
        # Key indices for eyes and mouth
        self.LEFT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.MOUTH_INDICES = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        
        # For head pose estimation
        self.FACE_3D_POINTS = np.array([
            [0.0, 0.0, 0.0],             # Nose tip
            [0.0, -330.0, -65.0],        # Chin
            [-225.0, 170.0, -135.0],     # Left eye left corner
            [225.0, 170.0, -135.0],      # Right eye right corner
            [-150.0, -150.0, -125.0],    # Left mouth corner
            [150.0, -150.0, -125.0]      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D facial landmark indices for head pose
        self.FACE_2D_INDICES = [1, 33, 61, 199, 291, 405]
    
    def detect_landmarks(self, image):
        """
        Detect facial landmarks in the image
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            landmarks: List of landmark coordinates (x, y, z) normalized to [0, 1]
            face_detected: Boolean indicating if face was detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        landmarks = None
        face_detected = False
        
        if results.multi_face_landmarks:
            face_detected = True
            # Get the first face (assuming single face detection)
            face_landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to numpy array
            h, w = image.shape[:2]
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.append([
                    landmark.x * w,  # x coordinate in pixels
                    landmark.y * h,  # y coordinate in pixels
                    landmark.z * w   # z coordinate (depth)
                ])
            landmarks = np.array(landmarks)
        
        return landmarks, face_detected, results
    
    def get_eye_landmarks(self, landmarks):
        """
        Extract eye landmark coordinates
        
        Args:
            landmarks: Full facial landmarks array
            
        Returns:
            left_eye: Left eye landmarks
            right_eye: Right eye landmarks
        """
        if landmarks is None:
            return None, None
        
        left_eye = landmarks[self.LEFT_EYE_INDICES]
        right_eye = landmarks[self.RIGHT_EYE_INDICES]
        
        return left_eye, right_eye
    
    def get_mouth_landmarks(self, landmarks):
        """
        Extract mouth landmark coordinates
        
        Args:
            landmarks: Full facial landmarks array
            
        Returns:
            mouth: Mouth landmarks
        """
        if landmarks is None:
            return None
        
        mouth = landmarks[self.MOUTH_INDICES]
        return mouth
    
    def draw_landmarks(self, image, results):
        """
        Draw facial landmarks on the image
        
        Args:
            image: Input image
            results: MediaPipe results object
            
        Returns:
            image: Image with landmarks drawn
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    None,
                    self.mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        return image


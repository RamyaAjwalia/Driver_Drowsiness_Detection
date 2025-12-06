# Driver Drowsiness and Distraction Detection System

A real-time computer vision system that detects driver drowsiness and distraction using facial landmark analysis. The system monitors eye closure, yawning, and head pose to identify signs of decreased alertness.

## Authors
- **Ramya Ajwalia** (ID: 1002213648)
- **Neel Katrodiya** (ID: 1002254987)

## Features

- **Real-time Detection**: Processes live camera feed in real-time
- **Multiple Indicators**: 
  - Eye Aspect Ratio (EAR) for drowsiness detection
  - Mouth Aspect Ratio (MAR) for yawning detection
  - Head pose estimation for distraction detection
- **Visual Alerts**: Clear on-screen warnings when drowsiness or distraction is detected
- **Performance Metrics**: Tracks and displays detection statistics
- **Video Processing**: Supports both live camera and video file input

## System Requirements

- Python 3.8 or higher
- Webcam or camera device
- Windows/Linux/macOS

## Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import cv2, mediapipe, numpy; print('All packages installed successfully')"
   ```

## Usage

### Real-time Detection (Live Camera)

Run the main application with default settings:
```bash
python src/main.py
```

**Command-line options**:
```bash
python src/main.py --camera 0 --ear-threshold 0.25 --mar-threshold 0.5 --save-output --output-path output
```

**Parameters**:
- `--camera`: Camera device index (default: 0)
- `--ear-threshold`: EAR threshold for drowsiness detection (default: 0.25, lower = more sensitive)
- `--mar-threshold`: MAR threshold for yawning detection (default: 0.5, higher = more sensitive)
- `--save-output`: Save output video to file
- `--output-path`: Directory to save output files (default: 'output')

**Controls**:
- Press `q` to quit
- Press `r` to reset statistics

### Video File Processing

Process a pre-recorded video file:
```bash
python src/test_video.py --video path/to/video.mp4 --output output_video.avi
```

**Parameters**:
- `--video`: Path to input video file (required)
- `--output`: Path to save processed output video (optional)
- `--ear-threshold`: EAR threshold (default: 0.25)
- `--mar-threshold`: MAR threshold (default: 0.5)

**Controls**:
- Press `q` to quit
- Press `p` to pause/resume

### Evaluation

Evaluate the system on a video with ground truth labels:
```bash
python src/evaluate.py --video path/to/video.mp4 --ground-truth ground_truth.json --output evaluation_results
```

**Ground Truth Format** (JSON):
```json
[
  {"frame": 0, "drowsy": false, "distracted": false},
  {"frame": 100, "drowsy": true, "distracted": false},
  {"frame": 200, "drowsy": false, "distracted": true}
]
```

## Project Structure

```
CV Project/
├── src/
│   ├── __init__.py
│   ├── facial_landmarks.py      # Facial landmark detection using MediaPipe
│   ├── feature_extraction.py    # EAR, MAR, and head pose calculation
│   ├── detector.py              # Drowsiness and distraction classification
│   ├── main.py                  # Main real-time application
│   ├── test_video.py            # Video file processing script
│   └── evaluate.py              # Evaluation and metrics computation
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Methodology

### 1. Facial Landmark Detection
- Uses MediaPipe Face Mesh for robust facial landmark detection
- Detects 468 facial landmarks in real-time
- Extracts key regions: eyes, mouth, and facial structure

### 2. Feature Extraction

**Eye Aspect Ratio (EAR)**:
- Measures eye opening/closing
- Formula: `EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)`
- Low EAR indicates closed eyes

**Mouth Aspect Ratio (MAR)**:
- Measures mouth opening (yawning)
- Formula: `MAR = (|p2-p8| + |p3-p7| + |p4-p6|) / (3 * |p1-p5|)`
- High MAR indicates yawning

**Head Pose Estimation**:
- Calculates pitch, yaw, and roll angles
- Uses solvePnP with 3D facial model
- Detects head deviations indicating distraction

### 3. Classification
- **Drowsiness**: Triggered when EAR is below threshold for consecutive frames
- **Distraction**: Triggered by yawning (high MAR) or head pose deviation
- Configurable thresholds and frame counts for tuning sensitivity

## Performance Metrics

The system tracks:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

## Advantages

- **High Societal Impact**: Contributes to safer roadways
- **Real-time Performance**: Efficient processing suitable for embedded systems
- **Non-intrusive**: No physical sensors required
- **Open Source**: Built with freely available tools

## Limitations

- **Environmental Sensitivity**: Performance may vary with lighting conditions
- **Occlusion**: Sunglasses or face coverings may hinder detection
- **Camera Quality**: Requires adequate camera resolution and frame rate
- **Generalization**: May need calibration for different demographics

## Datasets

The system is designed to work with:
- **NTHU Drowsy Driver Dataset**: Diversified recordings with multiple drowsiness levels
- **YawDD Dataset**: Focuses on yawning events and facial landmarks

## Future Improvements

- Integration with machine learning classifiers (SVM, neural networks)
- Multi-person detection support
- Mobile/embedded deployment optimization
- Integration with vehicle systems (ADAS)
- Improved handling of edge cases (sunglasses, low light)

## References

1. Real-time driver monitoring system with facial landmark-based eye analysis. Scientific Reports, Nature, 2023.
2. RealTime Drowsiness Detection System using Facial Landmarks. ACM Digital Library, 2024.
3. Real-Time Drowsiness Detection Using Eye Aspect Ratio and Facial Landmark Detection. arXiv, 2024.
4. A Review of Recent Developments in Driver Drowsiness Detection Systems. Medicina, 2022.
5. Driver Drowsiness Detection Using Facial Landmarks. Journal of Engineering Science, 2024.

## Troubleshooting

### Camera not detected
- Check camera permissions
- Try different camera indices: `--camera 1`, `--camera 2`, etc.
- Verify camera is not being used by another application

### Poor detection accuracy
- Adjust thresholds: Lower `--ear-threshold` for more sensitive drowsiness detection
- Ensure good lighting conditions
- Position camera to capture full face clearly

### Performance issues
- Reduce video resolution
- Close other applications
- Use GPU acceleration if available

## License

This project is developed for academic purposes as part of a Computer Vision course.

## Contact

For questions or issues, please contact:
- Ramya Ajwalia (ID: 1002213648)
- Neel Katrodiya (ID: 1002254987)


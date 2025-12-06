# Quick Start Guide

## Installation (5 minutes)

1. **Install Python 3.8+** (if not already installed)
   - Download from [python.org](https://www.python.org/downloads/)

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Running the System

### Option 1: Simple Launch (Recommended)
```bash
python run.py
```

### Option 2: Full Control
```bash
python src/main.py --camera 0
```

### Option 3: With Custom Thresholds
```bash
python src/main.py --ear-threshold 0.20 --mar-threshold 0.6
```

## Basic Usage

1. **Start the application** - The camera will open automatically
2. **Position yourself** - Sit in front of the camera with your face clearly visible
3. **Test detection**:
   - Close your eyes for 3+ seconds → Drowsiness alert
   - Yawn → Distraction alert
   - Turn your head significantly → Distraction alert
4. **Press 'q'** to quit

## Troubleshooting

**Camera not working?**
- Try: `python src/main.py --camera 1`
- Check camera permissions in system settings

**Detection not working?**
- Ensure good lighting
- Face the camera directly
- Adjust thresholds in `config.py` or via command line

**Performance issues?**
- Close other applications
- Reduce video resolution in camera settings

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Try processing video files: `python src/test_video.py --video your_video.mp4`
- Evaluate on datasets: `python src/evaluate.py --video dataset_video.mp4`


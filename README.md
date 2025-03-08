# Thermal Road User Detection System

A lightweight embedded system for detecting road users (cars, pedestrians, and cyclists) using thermal camera data.

## Requirements

- Python 3.8+
- Raspberry Pi (or similar embedded device)
- Network connection to thermal camera feed

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd thermal-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your thermal camera URL:
```bash
THERMAL_CAMERA_URL=http://your-thermal-camera-url
```

## Usage

Run the main script:
```bash
python main.py
```

The system will:
1. Connect to the thermal camera feed
2. Process incoming frames in real-time
3. Detect and classify road users
4. Display results with bounding boxes (if in debug mode)

Press 'q' to quit the application.

## Output Format

The system outputs detections in the following format:
- Class: Car/Pedestrian/Cyclist
- Confidence score: 0.0-1.0
- Bounding box coordinates: (x1, y1, x2, y2)

## TODO

- [ ] Implement model loading and weights
- [ ] Add specific preprocessing for thermal images
- [ ] Implement detection logic
- [ ] Add configuration file for model parameters
- [ ] Add error handling and recovery
- [ ] Implement result logging 
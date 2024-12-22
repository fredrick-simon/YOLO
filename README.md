
# YOLO Object Detection Implementation

This project uses YOLOv8 (You Only Look Once) for real-time object detection. The project consists of three Python scripts, each with a specific functionality:

1. **Yolo1.py**: Basic YOLO object detection on video input.
2. **Yolo2.py**: YOLO object detection with the ability to save detected objects' details to a text file.
3. **Yolo3.py**: YOLO object detection using your phone's camera as a webcam.

## Requirements

Before running the scripts, make sure you have the following installed:

-  not higher than Python 3.9
- `pip` for managing Python packages

### Install Dependencies
```bash
# Install required dependencies
pip install -r requirements.txt
```

## Script Overview

### 1. Yolo1.py

This script performs basic YOLO object detection on a video input (from webcam or video file). It will display the detected objects with bounding boxes and labels on the screen.

#### Usage:
```bash
python Yolo1.py
```

### 2. Yolo2.py

This script performs YOLO object detection and saves the detected objects' details (timestamp, class, track ID, bounding box, and confidence score) into a text file named `detected_objects.txt`.

#### Usage:
```bash
python Yolo2.py
```

The output will be saved in the `detected_objects.txt` file in the same directory.

### 3. Yolo3.py

This script allows you to use your **phone's camera as a webcam** for real-time object detection. It streams the camera feed from your phone using the **IP Webcam** app (Android) and performs YOLO detection on the video feed.

#### Steps to Set Up Phone Webcam:

1. **Install the IP Webcam app**:
   - Download and install the **IP Webcam** app from the [Google Play Store](https://play.google.com/store/apps/details?id=com.pas.webcam).
   - Open the app and start the camera.
   - The app will provide an IP address and port (e.g., `http://192.168.1.2:8080`).

2. **Connect your phone and computer to the same Wi-Fi network**.

3. **Update the `url` variable in the script**:
   - Replace the placeholder IP address and port in the script with the actual IP address and port provided by the IP Webcam app.
   - Example: 
     ```python
     url = "http://192.168.1.2:8080/video"
     ```

4. **Run the script**:
   ```bash
   python Yolo3.py
   ```

This will stream the camera feed from your phone and perform YOLO object detection.

## Saving Detected Objects

In **Yolo2.py**, the detected objects are saved in a text file (`detected_objects.txt`). Each entry includes:

- **Timestamp**: The time the object was detected.
- **Class**: The class of the detected object (e.g., person, car).
- **Track ID**: The unique track ID for the object.
- **Bounding Box**: The coordinates of the bounding box around the object.
- **Confidence**: The confidence score for the detection.

Example of a saved entry:
```
Timestamp: 2024-12-22 12:34:56, Class: person, Track ID: 1, Bounding Box: [100, 200, 300, 400], Confidence: 0.85
```

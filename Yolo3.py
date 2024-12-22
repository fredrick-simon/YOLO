import cv2
import json
from ultralytics import YOLO
import cvzone
import threading
import os
from datetime import datetime

# Create a lock for thread safety
tts_lock = threading.Lock()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load COCO class names
with open("coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv8 model
model = YOLO("yolo11s.pt")

# URL of the phone camera feed from IP Webcam
# Replace this with the actual IP address and port from your phone's IP Webcam app
url = "http://192.0.0.4:8080/video"  # Example URL

# Open the video capture (use phone's camera over the network)
cap = cv2.VideoCapture(url)

# Set to store already spoken track IDs to avoid repeating
spoken_ids = set()

# List to store detected objects and their timestamps
detected_objects = []

# Get the current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the detected objects text file in the script's directory
txt_file_path = os.path.join(script_dir, 'detected_objects.txt')

# Function to save detected objects to a file
def save_detected_objects(detected_objects):
    with open(txt_file_path, 'w') as f:
        for obj in detected_objects:
            f.write(f"Timestamp: {obj['timestamp']}, Class: {obj['class']}, Track ID: {obj['track_id']}, "
                    f"Bounding Box: {obj['bounding_box']}, Confidence: {obj['confidence']}\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        frame = cv2.resize(frame, (1020, 500))
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Check if there are any boxes in the results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
            
            # Dictionary to count classes based on track IDs for the current frame
            current_frame_counter = {}

            # Iterate through detected objects
            for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
                c = class_names[class_id]
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cvzone.putTextRect(frame, f'{track_id}', (x1, y2), 1, 1)
                cvzone.putTextRect(frame, f'{c}', (x1, y1), 1, 1)
                
                # Count the object only if it's a new detection
                if track_id not in spoken_ids:
                    spoken_ids.add(track_id)
                    
                    # Increment the count for the detected class
                    if c not in current_frame_counter:
                        current_frame_counter[c] = 0
                    current_frame_counter[c] += 1

                    # Save the detected object with the current timestamp
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    detected_objects.append({
                        'timestamp': timestamp,
                        'class': c,
                        'track_id': track_id,
                        'bounding_box': [x1, y1, x2, y2],
                        'confidence': conf
                    })

            # Announce the current counts for each detected class
            for class_name, count in current_frame_counter.items():
                if count > 0:  # Only announce if there are detected objects
                    count_text = f"{count} {class_name}" if count > 1 else f"One {class_name}"
                    # play_sound_async(count_text)  # Convert count to speech
                    current_frame_counter[class_name] = 0  # Reset count after announcement

        # Display the frame
        cv2.imshow("Phone Camera Feed", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("Script interrupted by user. Saving the file...")

# Save the detected objects to a file when the program ends or is interrupted
save_detected_objects(detected_objects)

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
from ultralytics import YOLO
from collections import deque

# Load YOLO model
model = YOLO("yolov8n.pt")

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load helmet reference image
helmet_reference = cv2.imread("helmet.png", cv2.IMREAD_GRAYSCALE)
helmet_reference = cv2.resize(helmet_reference, (100, 100))

# Initialize Tkinter
root = tk.Tk()
root.title("Helmet Detection System")
root.geometry("800x600")

# Video feed label
video_label = Label(root)
video_label.pack()

# Status label
status_label = Label(root, text="Press 'Start' to Detect", font=("Arial", 14))
status_label.pack()

# Open webcam
cap = cv2.VideoCapture(0)

# ORB feature matcher
orb = cv2.ORB_create()

# Use a queue to track last N detections for stability
detection_history = deque(maxlen=10)  # Stores last 10 detections


def detect_helmet():
    global detection_history

    ret, frame = cap.read()
    if not ret:
        return

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    helmet_detected = False  # Default state

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract face region
        face_region = frame[y:y + h, x:x + w]

        # Run YOLO object detection
        results = model(face_region)

        for obj in results[0].boxes.data:
            class_id = int(obj[5])
            confidence = float(obj[4])

            if class_id == 0 and confidence > 0.6:  # Confidence threshold
                helmet_detected = True
                break

        # Check with helmet reference using ORB Feature Matching
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (100, 100))

        kp1, des1 = orb.detectAndCompute(face_resized, None)
        kp2, des2 = orb.detectAndCompute(helmet_reference, None)

        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            if len(matches) > 15:  # Increased threshold for stability
                helmet_detected = True

    # Stability Fix: Use moving average of last 10 detections
    detection_history.append(helmet_detected)
    stable_output = sum(detection_history) > (len(detection_history) / 2)

    # Display Result
    if stable_output:
        status_label.config(text="Helmet Detected ✅", fg="green")
    else:
        status_label.config(text="No Helmet ❌", fg="red")

    # Convert frame for Tkinter display
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    # Refresh frame
    video_label.after(10, detect_helmet)


# Start Button
start_button = Button(root, text="Start Detection", command=detect_helmet, font=("Arial", 12))
start_button.pack()

# Exit Button
exit_button = Button(root, text="Exit", command=root.quit, font=("Arial", 12))
exit_button.pack()

# Run Tkinter loop
root.mainloop()

# Release webcam on exit
cap.release()
cv2.destroyAllWindows()

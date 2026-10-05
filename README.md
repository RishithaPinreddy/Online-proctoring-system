# SmartProctorX – Online Proctoring System
Online exam proctoring system for detecting multiple faces, mobile phones, and head movements using computer vision.

## Features
- **Head Pose Estimation** – Monitors head orientation and issues a warning when the head turns beyond the defined threshold.
- **Eye Gaze Tracking** – Tracks gaze direction and detects significant deviations.
- **Lip Movement Detection** – Monitors lip movements that may indicate talking or reading aloud.
- **Multiple-Person Detection** – Detects multiple people appearing in the camera frame.
- **Mobile/Object Detection** – Detects mobile phones and other unauthorized objects.
- **Real-Time Monitoring** – Integrates the detection modules into a live video stream with warnings for detected activities.

## Technologies Used
- Python
- OpenCV
- YOLO
- MediaPipe
- Computer Vision

## How It Works
The system captures the student's live camera feed and applies multiple computer-vision modules simultaneously. Each module analyzes a particular aspect of the video, such as head orientation, eye gaze, lip movement, people, or objects. Detected suspicious activities generate real-time warnings.

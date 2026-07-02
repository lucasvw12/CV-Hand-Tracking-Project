# Real-Time Hand Tracking and 3D Interaction System

A real-time computer vision project that uses MediaPipe and OpenCV to track hand movements from a webcam and enable intuitive 3D object manipulation in Unity. The system supports object movement, rotation, scaling, drawing, pinch-based interactions, and gesture recognition. Hand tracking data is transmitted over UDP, allowing the Unity scene to respond in real time with low-latency object manipulation and physics-based throwing.

## Features

- Real-time hand landmark tracking
- Hand gesture recognition
- 2d object manipulation
- Free-form drawing
- UDP communication between Python and Unity
- Multithreaded C# UDP receiver
- 3D object translation
- Quaternion-based object rotation
- Dynamic object scaling
- Velocity-based throwing using Unity physics

## Technologies

- Python
- OpenCV
- MediaPipe
- Unity
- C#
- UDP Networking

## How It Works

The Python application captures webcam frames using OpenCV and detects hand landmarks with MediaPipe. From these landmarks, the program calculates the hand's position, rotation, scale, and movement velocity. Aditionally, certain gestures such as pinching, pinky-up, fist up or pointing are classified and lead to different forms of 2d object manipulation on the webcam display. This information is packaged into a UDP message and sent to Unity.

On the Unity side, a background thread continuously listens for incoming UDP packets while the main thread renders the scene. The received data updates the object's position, rotation, scale, and applies velocity to simulate throwing, allowing for smooth real-time interaction.

## Project Structure

CV-Hand-Tracking-Project/
│
├── Assets/
│   └── Scripts/
│       └── UDPReceiver.cs
│
├── Packages/
├── ProjectSettings/
│
├── Python/
│   └── hand_tracking.py
│
├── requirements.txt
├── README.md
└── .gitignore


## Running the Project

1. Install the required Python packages.
```bash
pip install -r requirements.txt
```
2. Start the Python application.
```bash
python Python/hand_tracking.py
```
3. Open the Unity project.
4. Press Play in Unity.
   
## Future Improvements

- Support multiple hands
- Additional gesture recognition and features such as morphing
- Improved interaction mechanics and reduced latency
- Cross-device networking where other devices can connect to mine for object control. 
- Integration with VR/AR applications

## Author

Lucas Van Winckel

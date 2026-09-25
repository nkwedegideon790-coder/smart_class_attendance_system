# Smart Class Attendance System

A smart classroom attendance solution built with Python, Streamlit, OpenCV, and face-recognition technology. The system allows instructors to register students, capture face images, run live attendance sessions from uploaded video, and review attendance analytics in one simple dashboard.

## Overview

This project streamlines classroom attendance by combining:

- Face registration for new students
- Video-based attendance checking
- Matching against stored face embeddings
- SQLite storage for student records and attendance logs
- Attendance analytics for trends, participation, and absentee tracking

The app is designed to run as a desktop-friendly Streamlit web application and is suitable for classroom demos, academic projects, and lightweight attendance automation.

## Features

- Student registration with webcam or uploaded photographs
- Face detection and facial embedding generation
- Real-time attendance marking from video input
- Duplicate student handling and photo validation
- Dashboard for session history and attendance summaries
- Attendance rate tracking over time
- Identification of frequent absentees
- Local database storage with no external backend required

## Tech Stack

- Python 3.10+
- Streamlit
- OpenCV
- InsightFace
- OpenVINO
- NumPy
- SQLite
- Supervision

## Project Structure

```text
smart_class_attendance_system/
├── App.py                     # Main Streamlit entry point
├── analytics.py               # Attendance summaries and chart data
├── Database.py                # SQLite database setup and data helpers
├── models.py                  # Face model loading and embedding logic
├── attendance.db              # Local SQLite database (created at runtime)
├── Pages/
│   ├── Register_student.py    # Student registration page
│   └── live_attendance.py     # Live attendance session page
├── Readme.md                  # Project documentation
└── kernel.errors.txt          # Local error log / troubleshooting notes
```

## Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- A working webcam or video file for attendance testing
- A compatible NVIDIA GPU, Intel GPU, or CPU environment
- Internet access for the first-time model download if InsightFace assets are not already present

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd smart_class_attendance_system
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

3. Install the required dependencies:

```bash
pip install streamlit opencv-python numpy insightface openvino supervision
```

If you are using a different environment, install the packages from your preferred Python packaging setup as needed.

4. Ensure the InsightFace model files are available. The app expects the Buffalo_L model under the default InsightFace model folder, typically:

```text
~/.insightface/models/buffalo_l/
```

If the model is missing, the framework may download it automatically during the first run depending on your environment and setup.

## Running the Application

From the project root, start the app with:

```bash
streamlit run App.py
```

This launches the main dashboard. Use the sidebar navigation to switch between:

- Register Students
- Live Attendance
- Analytics

## How It Works

### 1. Register Students

The registration page allows you to add one or more student images. The app detects a face in each image, generates a facial embedding, and stores it in the local database.

### 2. Run Attendance Session

The live attendance page processes uploaded video frames, detects faces, tracks them over time, and compares each detected face embedding against the known student embeddings. If a match is found, the student is marked as present.

### 3. Review Attendance Analytics

The analytics panel displays:

- Daily attendance summary
- Attendance rate by student
- Session history
- Attendance trend over recent days
- Frequent absentees

## Database Design

The project uses SQLite with tables for:

- Students
- Face image samples with embeddings
- Attendance records

This keeps the workflow simple and local while preserving enough data for attendance tracking and reporting.

## Usage Example

1. Open the app with Streamlit.
2. Go to the Registration page.
3. Add a student name and upload 2–3 face photos from different angles.
4. Save the student.
5. Open the Live Attendance page.
6. Upload a classroom video or run detection against a supported input.
7. Review attendance results and analytics from the dashboard.

## Notes

- The project is designed for lightweight local deployment and educational use.
- Model accuracy depends on image quality, lighting, and face orientation.
- It is best to enroll students with multiple clear facial photos for stronger recognition performance.
- The app stores attendance locally using SQLite, making it easy to test without a remote server.

## Troubleshooting

### No face detected

- Use better lighting
- Ensure the face is clearly visible
- Avoid multiple people in a single image

### Recognition mismatches

- Add more photos per student
- Capture images from slightly different angles
- Use consistent frontal or near-frontal views

### Model load issues

- Confirm your Python environment has the required dependencies installed
- Check that the InsightFace buffalo_l model files exist
- Reinstall the relevant packages if the model path is missing or corrupted

## Future Improvements

Potential enhancements include:

- Real-time webcam attendance capture
- Better false-positive filtering for unknown faces
- Admin login and role-based access
- Exporting attendance records as CSV or Excel
- Better UI refinements and detailed reporting
- Support for multi-classroom management

## Summary

The Smart Class Attendance System is a practical, AI-assisted way to automate classroom attendance using face recognition. It combines registration, recognition, and reporting in a single easy-to-use Streamlit app, making it ideal for educational projects and classroom prototypes.

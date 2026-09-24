import datetime
import cv2
import numpy as np
import supervision as sv
import streamlit as st
import tempfile
from models import load_models, get_embedding_ov
from Database import init_db, get_all_students_for_matching
from analytics import (
    get_attendance_summary, get_attendance_rate_by_student,
    get_frequent_absentees, get_attendance_trend
)

# Streamlit page configuration
st.set_page_config(page_title="Smart Attendance System", layout="wide")
st.title("Smart Attendance System")

# Load models and initialize database
face_app, rec_model, rec_output = load_models()
conn = init_db()
known_students = get_all_students_for_matching(conn)  # [(id, name, embedding), ...]

marked_present = set()
track_to_student = {}      # track_id -> (student_id, name)
present_students = {}      # student_id -> {"name": name, "image": crop}
FRAME_SKIP = 5
frame_count = 0

# Function to match a student's embedding with known students
def match_student(embedding, known_students, threshold=0.6):
    best_match, best_score = None, -1
    for student_id, name, known_emb in known_students:
        score = np.dot(embedding, known_emb)
        if score > best_score:
            best_match, best_score = (student_id, name), score
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score

# Streamlit file uploader for video input
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    tracker = sv.ByteTrack()
    # 
    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.subheader("Live feed")
        frame_placeholder = st.empty()
    with right_col:
        st.subheader("Attendance")
        table_placeholder = st.empty()
    #  
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # 
        if frame_count % FRAME_SKIP == 0:
            faces = face_app.get(frame)

            if len(faces) > 0:
                xyxy = np.array([f.bbox for f in faces], dtype=float)
                confidence = np.array([f.det_score for f in faces], dtype=float)
                class_id = np.zeros(len(faces), dtype=int)
                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)

            for i in range(len(tracked)):
                x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                track_id = tracked.tracker_id[i]

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                if track_id in track_to_student:
                    student_id, name = track_to_student[track_id]
                else:
                    embedding = get_embedding_ov(rec_model, rec_output, crop)
                    match, score = match_student(embedding, known_students)
                    print(f"Loaded {len(known_students)} known students for matching")
                    if match:
                        student_id, name = match
                        track_to_student[track_id] = (student_id, name)
                    else:
                        student_id, name = None, "Unknown"

                if student_id is not None:
                    color, label = (0, 200, 0), name

                    if student_id not in marked_present:
                        marked_present.add(student_id)
                        present_students[student_id] = {"name": name, "image": crop.copy()}

                        now = datetime.datetime.now()
                        conn.execute(
                            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
                            (student_id, now.date().isoformat(), now.time().isoformat(timespec="seconds"), "Present")
                        )
                        conn.commit()
                else:
                    color, label = (0, 0, 255), "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            with table_placeholder.container():
                st.write(f"**Present ({len(present_students)})**")
                cols_per_row = 3
                items = list(present_students.items())
                for i in range(0, len(items), cols_per_row):
                    row_items = items[i:i + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, (student_id, info) in zip(cols, row_items):
                        with col:
                            img_rgb = cv2.cvtColor(info["image"], cv2.COLOR_BGR2RGB)
                            st.image(img_rgb, caption=info["name"], width=100)

    cap.release()
    st.success("Done processing video.")

st.write("Attendance Analytics")

summary = get_attendance_summary(conn)
col1, col2, col3 = st.columns(3)
col1.metric("Present today", summary["present"])
col2.metric("Absent today", summary["absent"])
col3.metric("Attendance rate", f"{summary['attendance_rate']}%")

st.subheader("Attendance trend")
trend = get_attendance_trend(conn)
if trend:
    st.line_chart({d["date"]: d["rate"] for d in trend})
else:
    st.write("No attendance data yet.")

st.subheader("Frequent absentees (below 75%, last 30 days)")
absentees = get_frequent_absentees(conn)
if absentees:
    for a in absentees:
        st.write(f"**{a['name']}** — {a['attendance_rate']}% ({a['days_present']}/{a['total_sessions']} sessions)")
else:
    st.write("No students below the threshold.")

st.subheader("Full class breakdown")
rates = get_attendance_rate_by_student(conn)
st.dataframe(rates)
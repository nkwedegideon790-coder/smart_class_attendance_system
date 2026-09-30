import datetime
import tempfile

import cv2
import numpy as np
import streamlit as st
import supervision as sv

from models import load_models, get_embedding_ov
from Database import init_db, get_all_students_for_matching
from analytics import (
    get_attendance_summary, get_attendance_rate_by_student,
    get_frequent_absentees, get_attendance_trend,
    get_session_history, get_session_detail,
)

st.title("Smart Attendance System")

face_app = load_models()
conn = init_db()
known_students = get_all_students_for_matching(conn)  # [(id, name, embedding), ...]

FRAME_SKIP = 5
THRESHOLD = 0.6

# ---------- state that survives page switches ----------
if "last_frame" not in st.session_state:
    st.session_state.last_frame = None
if "present_students" not in st.session_state:
    st.session_state.present_students = {}
if "processed_file" not in st.session_state:
    st.session_state.processed_file = None


# ---------- helpers ----------
def match_student(embedding, known_students, threshold=THRESHOLD):
    best_match, best_score = None, -1
    for student_id, name, known_emb in known_students:
        score = np.dot(embedding, known_emb)
        if score > best_score:
            best_match, best_score = (student_id, name), score
    if best_score >= threshold:
        return best_match, best_score
    return None, best_score


def make_square(img: np.ndarray, size: int = 120) -> np.ndarray:
    h, w = img.shape[:2]
    min_dim = min(h, w)
    top = (h - min_dim) // 2
    left = (w - min_dim) // 2
    return cv2.resize(img[top:top + min_dim, left:left + min_dim], (size, size))


def render_present(container, students):
    with container.container():
        st.write(f"**Present ({len(students)})**")
        cols_per_row = 3
        items = list(students.items())
        for i in range(0, len(items), cols_per_row):
            cols = st.columns(cols_per_row, gap="medium")
            for col, (sid, info) in zip(cols, items[i:i + cols_per_row]):
                with col:
                    img = cv2.cvtColor(make_square(info["image"]), cv2.COLOR_BGR2RGB)
                    st.image(img, caption=info["name"], width=120)
            st.write("")


# ---------- layout ----------
uploaded_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

left_col, right_col = st.columns([2, 1])
with left_col:
    st.subheader("Live feed")
    frame_placeholder = st.empty()
with right_col:
    st.subheader("Attendance")
    table_placeholder = st.empty()

if st.session_state.last_frame is not None:
    frame_placeholder.image(st.session_state.last_frame, channels="RGB")
    render_present(table_placeholder, st.session_state.present_students)

# ---------- process video (only once per uploaded file) ----------
if uploaded_file is not None:
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.processed_file != file_key:
        st.session_state.processed_file = file_key
        st.session_state.present_students = {}
        st.session_state.last_frame = None

        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        tracker = sv.ByteTrack()

        marked_present = set()
        track_to_student = {}
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            faces = face_app.get(frame)  # each face already has .bbox, .det_score, .embedding

            if len(faces) > 0:
                xyxy = np.array([f.bbox for f in faces], dtype=float)
                confidence = np.array([f.det_score for f in faces], dtype=float)
                class_id = np.zeros(len(faces), dtype=int)
                detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)

            # tracker.update_with_detections reorders/filters detections, so match
            # each tracked box back to its source face by nearest bbox rather than
            # assuming index alignment with `faces`
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
                    # find the source face whose bbox matches this tracked box
                    tracked_box = tracked.xyxy[i]
                    matched_face = min(
                        faces,
                        key=lambda f: np.sum((f.bbox - tracked_box) ** 2)
                    )
                    embedding = get_embedding_ov(matched_face)
                    match, score = match_student(embedding, known_students)
                    if match:
                        student_id, name = match
                        track_to_student[track_id] = (student_id, name)
                    else:
                        student_id, name = None, "Unknown"

                if student_id is not None:
                    color, label = (0, 200, 0), name

                    if student_id not in marked_present:
                        marked_present.add(student_id)
                        st.session_state.present_students[student_id] = {
                            "name": name, "image": crop.copy()
                        }
                        now = datetime.datetime.now()
                        conn.execute(
                            "INSERT INTO attendance (student_id, date, time, status) VALUES (?, ?, ?, ?)",
                            (student_id, now.date().isoformat(),
                             now.time().isoformat(timespec="seconds"), "Present"),
                        )
                        conn.commit()
                else:
                    color, label = (0, 0, 255), "Unknown"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.last_frame = frame_rgb
            frame_placeholder.image(frame_rgb, channels="RGB")
            render_present(table_placeholder, st.session_state.present_students)

        cap.release()
        st.success("Done processing video.")

# ---------- analytics ----------
st.divider()
st.header("Attendance Analytics")

summary = get_attendance_summary()
c1, c2, c3 = st.columns(3)
c1.metric("Present today", summary["present"])
c2.metric("Absent today", summary["absent"])
c3.metric("Attendance rate", f"{summary['attendance_rate']}%")

st.subheader("Session history")
sessions = get_session_history()
if not sessions:
    st.write("No sessions recorded yet.")
else:
    for s in sessions:
        with st.expander(
            f"{s['date']} — {s['present_count']}/{s['total_students']} present "
            f"({s['attendance_rate']}%)"
        ):
            st.write(f"First marked: {s['first_marked']} · Last marked: {s['last_marked']}")
            for name, time in get_session_detail(s["date"]):
                st.write(f"- {name} — {time}")

st.subheader("Attendance trend")
trend = get_attendance_trend()
if trend:
    st.line_chart({d["date"]: d["rate"] for d in trend})
else:
    st.write("No attendance data yet.")

st.subheader("Frequent absentees (below 75%, last 30 days)")
absentees = get_frequent_absentees()
if absentees:
    for a in absentees:
        st.write(f"**{a['name']}** — {a['attendance_rate']}% "
                 f"({a['days_present']}/{a['total_sessions']} sessions)")
else:
    st.write("No students below the threshold.")

st.subheader("Full class breakdown")
st.dataframe(get_attendance_rate_by_student())
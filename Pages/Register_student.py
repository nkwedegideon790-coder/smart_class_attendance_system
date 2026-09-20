import streamlit as st
import cv2
import numpy as np
from models import load_models, get_face_crop_and_embedding
from Database import init_db, add_student, add_image, get_all_students_for_matching

st.set_page_config(page_title="Register Students", layout="wide")
st.title("Register Students")

face_app, rec_model, rec_output = load_models()
conn = init_db()

# ---------- Add new student ----------
st.subheader("Add a new student")

name = st.text_input("Student name")
capture_mode = st.radio("How do you want to add photos?", ["Webcam", "Upload photo(s)"], horizontal=True)

captured = []  # list of (crop, embedding) pairs collected this session

if capture_mode == "Webcam":
    img_file = st.camera_input("Take a photo")
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = get_face_crop_and_embedding(face_app, rec_model, rec_output, frame)

        if len(results) == 0:
            st.warning("No face detected — try again with better lighting.")
        elif len(results) > 1:
            st.warning("Multiple faces detected — make sure only one student is in frame.")
        else:
            crop, embedding = results[0]
            captured.append((crop, embedding))
            st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=150, caption="Captured")

else:
    uploaded = st.file_uploader(
        "Upload one or more clear photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )
    if uploaded:
        cols = st.columns(min(len(uploaded), 4))
        for i, f in enumerate(uploaded):
            file_bytes = np.frombuffer(f.getvalue(), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, dtype=np.uint8) if False else cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            results = get_face_crop_and_embedding(face_app, rec_model, rec_output, frame)

            with cols[i % 4]:
                if len(results) == 1:
                    crop, embedding = results[0]
                    captured.append((crop, embedding))
                    st.image(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), width=120, caption=f"{f.name} ✓")
                else:
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=120,
                              caption=f"{f.name} ✗ ({len(results)} faces)")

if captured:
    st.success(f"{len(captured)} valid face photo(s) ready to save")

if st.button("Save Student", type="primary", disabled=(not name or not captured)):
    student_id = add_student(conn, name)
    for crop, embedding in captured:
        add_image(conn, student_id, crop, embedding)
    st.success(f"Registered {name} with {len(captured)} photo(s)")
    st.rerun()

st.divider()

# ---------- View / manage existing students ----------
st.subheader("Registered students")

rows = conn.execute("""
    SELECT s.id, s.name, COUNT(i.id) as photo_count, MIN(i.image) as sample_image
    FROM students s
    LEFT JOIN images i ON i.student_id = s.id
    GROUP BY s.id, s.name
    ORDER BY s.name
""").fetchall()

if not rows:
    st.write("No students registered yet.")
else:
    cols_per_row = 4
    for i in range(0, len(rows), cols_per_row):
        row_chunk = rows[i:i + cols_per_row]
        cols = st.columns(cols_per_row)
        for col, (student_id, sname, photo_count, sample_image) in zip(cols, row_chunk):
            with col:
                if sample_image is not None:
                    img = cv2.imdecode(np.frombuffer(sample_image, dtype=np.uint8), cv2.IMREAD_COLOR)
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=120)
                st.write(f"**{sname}**")
                st.caption(f"{photo_count} photo(s)")
                if st.button("Delete", key=f"del_{student_id}"):
                    conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
                    conn.commit()
                    st.rerun()
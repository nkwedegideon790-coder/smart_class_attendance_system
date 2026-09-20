import sqlite3
import cv2
import numpy as np


def init_db(db_path="attendance.db"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            image BLOB NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        )
    """)
    conn.commit()
    return conn


def add_student(conn, name: str) -> int:
    cursor = conn.execute("INSERT INTO students (name) VALUES (?)", (name,))
    conn.commit()
    return cursor.lastrowid


def add_image(conn, student_id: int, face_crop: np.ndarray, embedding: np.ndarray):
    success, encoded_img = cv2.imencode(".jpg", face_crop)
    if not success:
        raise ValueError("Failed to encode image")
    conn.execute(
        "INSERT INTO images (student_id, image, embedding) VALUES (?, ?, ?)",
        (student_id, encoded_img.tobytes(), embedding.astype(np.float32).tobytes())
    )
    conn.commit()


def get_all_students_for_matching(conn):
    rows = conn.execute("""
        SELECT s.id, s.name, i.embedding FROM students s
        JOIN images i ON i.student_id = s.id
    """).fetchall()

    by_student = {}
    for student_id, name, embedding_bytes in rows:
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        by_student.setdefault((student_id, name), []).append(embedding)

    matching_set = []
    for (student_id, name), embeddings in by_student.items():
        avg = np.mean(embeddings, axis=0)
        avg = avg / np.linalg.norm(avg)
        matching_set.append((student_id, name, avg))
    return matching_set


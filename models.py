import cv2
import numpy as np
import streamlit as st
from insightface.app import FaceAnalysis


@st.cache_resource
def load_models():
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1)  # CPU — deterministic across all machines
    return face_app


def get_embedding_ov(face):
    """face is one result from face_app.get(frame) — InsightFace already computed the embedding."""
    return face.embedding / np.linalg.norm(face.embedding)


def get_face_crop_and_embedding(face_app, frame: np.ndarray):
    """
    Runs detection on a frame and returns a list of (face_crop, embedding) pairs —
    one per face found.
    """
    faces = face_app.get(frame)
    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        embedding = get_embedding_ov(face)
        results.append((crop, embedding))
    return results
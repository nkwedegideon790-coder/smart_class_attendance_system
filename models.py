import cv2
import numpy as np
import openvino as ov
from pathlib import Path
from insightface.app import FaceAnalysis
import streamlit as st

@st.cache_resource
def load_models():
    core = ov.Core()

    def pick_device(prefer=("GPU", "NPU", "CPU")):
        available = core.available_devices
        for dev in prefer:
            if any(d.startswith(dev) for d in available):
                return next(d for d in available if d.startswith(dev))
        return "CPU"

    device = pick_device()
    insightface_home = Path.home() / ".insightface" / "models" / "buffalo_l"
    rec_ir = insightface_home / "w600k_r50.xml"
    rec_model = core.compile_model(str(rec_ir), device, config={"PERFORMANCE_HINT": "LATENCY"})
    rec_output = rec_model.output(0)

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1)

    return face_app, rec_model, rec_output


def get_embedding_ov(rec_model, rec_output, face_crop):
    blob = cv2.dnn.blobFromImage(
        face_crop, scalefactor=1.0 / 127.5, size=(112, 112),
        mean=(127.5, 127.5, 127.5), swapRB=True
    )
    result = rec_model([blob])[rec_output]
    embedding = result.flatten()
    return embedding / np.linalg.norm(embedding)


def get_face_crop_and_embedding(face_app, rec_model, rec_output, frame: np.ndarray):
    faces = face_app.get(frame)
    results = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        embedding = get_embedding_ov(rec_model, rec_output, crop)
        results.append((crop, embedding))
    return results
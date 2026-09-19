import cv2
from ultralytics import YOLO
import supervision as sv
import openvino as ov
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
from insightface.utils import face_align


def convert_to_ir(onnx_path: str, ir_path: str):
    onnx_path, ir_path = Path(onnx_path), Path(ir_path)
    if ir_path.exists():
        return ir_path
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(ir_path))
    return ir_path


def pick_device(core: ov.Core, prefer=("GPU", "NPU", "CPU")) -> str:
    available = core.available_devices
    for dev in prefer:
        if any(d.startswith(dev) for d in available):
            return next(d for d in available if d.startswith(dev))
    return "CPU"


core = ov.Core()
print("Available devices:", core.available_devices)
device = pick_device(core)
print(f"Using device: {device}")

# --- YOLO (person detection) ---
yolo_ir = convert_to_ir("yolov8n.onnx", "yolov8n.xml")
yolo_model = core.compile_model(str(yolo_ir), device, config={"PERFORMANCE_HINT": "LATENCY"})
yolo_output = yolo_model.output(0)

# --- Locate InsightFace's downloaded ONNX models ---
# Trigger the download once first if this folder doesn't exist yet:
#   FaceAnalysis(name="buffalo_l").prepare(ctx_id=-1)
insightface_home = Path.home() / ".insightface" / "models" / "buffalo_l"

det_onnx = next(insightface_home.glob("det_*.onnx"))   # SCRFD detector
rec_onnx = next(insightface_home.glob("w600k_r50.onnx"))  # ArcFace recognizer

# --- SCRFD (face detection) via OpenVINO ---
det_ir = convert_to_ir(str(det_onnx), str(insightface_home / f"{det_onnx.stem}.xml"))
det_model = core.compile_model(str(det_ir), device, config={"PERFORMANCE_HINT": "LATENCY"})

# --- ArcFace (recognition) via OpenVINO ---
rec_ir = convert_to_ir(str(rec_onnx), str(insightface_home / "w600k_r50.xml"))
rec_model = core.compile_model(str(rec_ir), device, config={"PERFORMANCE_HINT": "LATENCY"})
rec_output = rec_model.output(0)

print(f"SCRFD + ArcFace compiled on {device}")


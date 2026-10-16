"""RF-DETR (Apache-2.0, COCO-pretrained) person / ball detector via ONNX Runtime.

The ONNX graph (exported by tools/export_rfdetr.py) takes a fixed (H, W) RGB image
normalized with ImageNet mean/std and returns 300 query boxes (cx, cy, w, h normalized)
and 91-way COCO logits. Post-processing follows RF-DETR: sigmoid scores per class.

Execution providers: "cpu" (reliable baseline) or "dml" (DirectML on the AMD iGPU);
benchmark both, do not assume DirectML is faster.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

COCO_PERSON = 1
COCO_SPORTS_BALL = 37
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) x1, y1, x2, y2 in source pixels
    scores: np.ndarray  # (N,)
    classes: np.ndarray  # (N,) COCO ids


class RFDETRDetector:
    def __init__(self, onnx_path: str | Path, provider: str = "cpu", threads: int = 0):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        prov = {"cpu": ["CPUExecutionProvider"],
                "dml": ["DmlExecutionProvider", "CPUExecutionProvider"]}[provider]
        if provider == "dml":
            so.enable_mem_pattern = False  # required by the DirectML EP
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(onnx_path), sess_options=so, providers=prov)
        self.provider = self.sess.get_providers()[0]
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.in_h, self.in_w = int(inp.shape[2]), int(inp.shape[3])

    def preprocess(self, bgr: np.ndarray) -> np.ndarray:
        img = cv2.resize(bgr, (self.in_w, self.in_h), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        return np.ascontiguousarray(img.transpose(2, 0, 1)[None])

    def __call__(self, bgr: np.ndarray, classes: tuple[int, ...] = (COCO_PERSON, COCO_SPORTS_BALL),
                 score_thresh: float = 0.1) -> Detections:
        h, w = bgr.shape[:2]
        dets, logits = self.sess.run(None, {self.input_name: self.preprocess(bgr)})
        prob = 1.0 / (1.0 + np.exp(-logits[0]))  # (300, 91)
        sel = prob[:, list(classes)]
        k = sel.argmax(1)
        s = sel[np.arange(len(sel)), k]
        keep = s >= score_thresh
        b = dets[0][keep]
        cx, cy, bw, bh = b[:, 0] * w, b[:, 1] * h, b[:, 2] * w, b[:, 3] * h
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        cls = np.asarray(classes)[k[keep]]
        return Detections(boxes, s[keep], cls)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.7) -> np.ndarray:
    """Indices kept by greedy NMS (DETR outputs are mostly duplicate-free; this is a
    safety net for the low-score tail)."""
    if len(boxes) == 0:
        return np.zeros(0, int)
    idx = cv2.dnn.NMSBoxes(
        [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes],
        scores.astype(float).tolist(), 0.0, iou)
    return np.asarray(idx, int).reshape(-1)

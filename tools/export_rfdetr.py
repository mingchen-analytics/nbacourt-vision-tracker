"""Download COCO-pretrained RF-DETR weights (Apache-2.0, sizes N/S/M only) and export ONNX.

Weights are cached under models/rfdetr (D: drive) via RF_HOME, not the default ~/.roboflow.

Usage: python tools/export_rfdetr.py --size small --shape 512 896
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=["nano", "small", "medium"], default="small")
    ap.add_argument("--shape", type=int, nargs=2, metavar=("H", "W"), required=True)
    a = ap.parse_args()
    os.environ.setdefault("RF_HOME", str(ROOT / "models" / "rfdetr"))
    import rfdetr

    cls = {"nano": rfdetr.RFDETRNano, "small": rfdetr.RFDETRSmall, "medium": rfdetr.RFDETRMedium}[a.size]
    model = cls()
    out = ROOT / "models" / "onnx" / f"rfdetr_{a.size}_{a.shape[0]}x{a.shape[1]}"
    out.mkdir(parents=True, exist_ok=True)
    path = model.export(output_dir=str(out), shape=tuple(a.shape), opset_version=17, verbose=False)
    print("exported", path)


if __name__ == "__main__":
    main()

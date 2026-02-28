"""
Standalone runner for PaddleOCR detection. Used in a subprocess so Paddle and PyTorch
never load in the same process (avoids "_gpuDeviceProperties is already registered").
Reads image path and params JSON path from argv; writes result JSON to stdout.
Usage: python -m modules.textdetector.paddle_det_runner <image_path> <params_json_path>
"""
import os
import sys
import json
import numpy as np
import cv2

# Must set before any paddle import
os.environ.setdefault("FLAGS_use_mkldnn", "0")
_path_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("PPOCR_HOME", os.path.join(_path_root, "data", "models", "paddle-ocr"))
if _path_root not in sys.path:
    sys.path.insert(0, _path_root)

def main():
    # Import paddle first so it is in sys.modules before PaddleOCR/PaddleX load (avoids ModuleNotFoundError in subprocess)
    import paddle  # noqa: F401

    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: paddle_det_runner <image_path> <params_json_path>"}), flush=True)
        sys.exit(1)
    image_path = sys.argv[1]
    params_path = sys.argv[2]
    if not os.path.isfile(image_path):
        print(json.dumps({"error": f"Image not found: {image_path}"}), flush=True)
        sys.exit(1)
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        print(json.dumps({"error": f"Failed to read image: {image_path}"}), flush=True)
        sys.exit(1)
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    from paddleocr import PaddleOCR

    det_side = int(params.get("det_limit_side_len", 960))
    det_thresh = float(params.get("det_db_thresh", 0.3))
    det_box_thresh = float(params.get("det_db_box_thresh", 0.6))
    model = PaddleOCR(
        use_textline_orientation=False,
        lang=params.get("lang", "ch"),
        device="cpu",
        enable_mkldnn=False,
        text_det_limit_side_len=det_side,
        text_det_thresh=det_thresh,
        text_det_box_thresh=det_box_thresh,
    )
    result = model.predict(
        img,
        use_textline_orientation=False,
        text_det_limit_side_len=det_side,
        text_det_thresh=det_thresh,
        text_det_box_thresh=det_box_thresh,
    )
    blocks = []
    if result and len(result) > 0:
        page = result[0]
        polys = page.get("rec_polys", [])
        if hasattr(polys, "tolist"):
            polys = polys.tolist()
        for box in polys or []:
            if box is None or (hasattr(box, "__len__") and len(box) < 4):
                continue
            pts = np.array(box, dtype=np.int32)
            if pts.ndim == 1:
                pts = pts.reshape(-1, 2)
            if pts.shape[0] < 4:
                continue
            x1 = int(pts[:, 0].min())
            y1 = int(pts[:, 1].min())
            x2 = int(pts[:, 0].max())
            y2 = int(pts[:, 1].max())
            if x2 <= x1 or y2 <= y1:
                continue
            blocks.append({
                "xyxy": [x1, y1, x2, y2],
                "lines": [pts.tolist()],
                "font_size": max(y2 - y1, 12),
            })
    print(json.dumps({"blocks": blocks}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

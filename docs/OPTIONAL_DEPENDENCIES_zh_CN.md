# 可选模块依赖与冲突说明（中文）

部分可选检测器/修复器的依赖版本与主 `requirements.txt` 不一致。本文给出冲突点与可行方案。

---

## 1) CRAFT 检测器（`craft_det`）

**冲突来源：** `craft-text-detector`（0.4.3）要求：
- `opencv-python>=3.4.8.29,<4.5.4.62`

而主项目通常使用：
- `opencv-python>=4.8.1.78`

因此默认环境下二者冲突。

**表现：**
- pip 可能报依赖冲突；
- 即使安装成功，升级 OpenCV 后 `craft_det` 可能在导入或推理时失败。

**建议方案：**
1. 不使用 `craft_det`：改用 `easyocr_det` 或 `mmocr_det`。
2. 单独虚拟环境：为 CRAFT 建专用 venv，安装旧 OpenCV + craft。
3. 保持主环境不装 craft：UI 可见但选择时会报错，改用其他检测器。

---

## 2) Simple LaMa 修复器（`simple_lama`）

**冲突来源：** `simple-lama-inpainting`（0.1.2）要求：
- `pillow>=9.5.0,<10.0.0`

而主项目通常是 Pillow 10.x。

**表现：**
- pip 依赖冲突；
- 在 Pillow 10.x 下可能导入失败或运行失败。

**建议方案：**
1. 不使用 `simple_lama`：改用 `lama_large_512px`、`lama_mpe`、`lama_onnx`、`lama_manga_onnx`。
2. 若必须用 `simple_lama`，降级 Pillow 到 `<10`（可能影响其他功能）。
3. 使用专用 venv，仅用于 `simple_lama`。

---

## 总结表

| 可选模块 | 冲突包 | 冲突点 | 推荐替代 |
|---|---|---|---|
| `craft_det` | craft-text-detector | opencv `<4.5.4.62` vs `>=4.8` | easyocr_det / mmocr_det / ctd |
| `simple_lama` | simple-lama-inpainting | pillow `<10` vs `10.x` | lama_large_512px / lama_onnx / lama_manga_onnx |
| `rapidocr_det` / `rapidocr` | 无 | 无 | 可选安装 `rapidocr-onnxruntime` |
| `nemotron_ocr_v1` | 无 | 仅支持 Python 3.12+ | 可选安装 `nemotron-ocr` |
| `nemotron_parse` | 无 | 需额外依赖 | 安装 `transformers accelerate albumentations timm` |

---

## 3) RapidOCR（`rapidocr_det`, `rapidocr`）

**无冲突**，可选安装：
- `rapidocr-onnxruntime`

用途：提供 ONNX 的文本检测与 OCR（CPU 可用），适合中/韩/英漫画场景。

---

## 4) Nemotron OCR v1（`nemotron_ocr_v1`）

可选包：`nemotron-ocr`。  
特点：整页检测+识别（NVIDIA），再按框重叠分配文本到项目文本块。  
要求：Python 3.12/3.13（不支持 3.10/3.11）。

---

## 5) Nemotron Parse 1.1（`nemotron_parse`）

可选依赖：`transformers`、`accelerate`、`torch`、`albumentations`、`timm`。  
特点：整页文档 OCR（带 bbox 与语义类别），更偏文档场景。


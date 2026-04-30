# 可选文本检测器与「检测+OCR」组合说明（中文）

## 哪些文本检测模型最适合搭配 **none_ocr**？

**none_ocr** 表示“对检测框不再执行 OCR”；程序会保留检测器已经写入到文本块中的内容。  
因此，**none_ocr** 只适用于“检测+识别一体（spotter）”并能写入 `blk.text` 的检测器。

| 检测器 | 会填充文本？ | 适合搭配 none_ocr？ |
|---|---|---|
| **stariver_ocr** | ✅ 是（API 返回框+文本） | ✅ **最佳**（一次 API 调用走完） |
| **hunyuan_ocr_det** | ✅ 是（整图 spotting） | ✅ **最佳** |
| **swintextspotter_v2** | ✅ 是（demo 输出含文本时） | ✅ 可用 |
| paddle_det_v5 / paddle_det / surya_det / easyocr_det / ctd / dptext_detr / mmocr_det 等 | ❌ 否（仅检测） | ❌ 文本会空，必须配 OCR |

**建议：**  
如果你想“只检测，不走 OCR 步骤”，请使用 spotter：`stariver_ocr` / `hunyuan_ocr_det` / `swintextspotter_v2`，并把 OCR 设为 `none_ocr`。  
若使用纯检测模型，务必再选一个 OCR（如 `paddle_rec_v5`、`surya_ocr`、`easyocr_ocr`、`mmocr_ocr`）。

---

## 检测与 OCR 覆盖（速览）

| 模块/家族 | 检测 | OCR | 说明 |
|---|---|---|---|
| **HunyuanOCR** | ✅ `hunyuan_ocr_det` | ✅ `hunyuan_ocr` | 可 `hunyuan_ocr_det + none_ocr`，也可再用 `hunyuan_ocr` 复识别 |
| **PaddleOCR v5** | ✅ `paddle_det_v5` | ✅ `paddle_rec_v5` | 推荐成对使用 |
| **PaddleOCR (full)** | ✅ `paddle_det` | ✅ `paddle_ocr` | 全流程可用 |
| **Surya** | ✅ `surya_det` | ✅ `surya_ocr` | 检测+裁剪识别 |
| **EasyOCR** | ✅ `easyocr_det` | ✅ `easyocr_ocr` | 检测+裁剪识别 |
| **Stariver (API)** | ✅ `stariver_ocr` | ✅ `stariver_ocr` | 检测器会填文字，适合 `none_ocr` |
| **SwinTextSpotter v2** | ✅ `swintextspotter_v2` | ✅ demo 有文本时可用 | 需设置 `repo_path` |
| **DPText-DETR** | ✅ `dptext_detr` | ❌ | 仅检测，需配 OCR |
| **MMOCR** | ✅ `mmocr_det` | ✅ `mmocr_ocr` | 依赖同一套 OpenMMLab |
| **Ocean-OCR** | ❌ | ✅ `ocean_ocr` | 仅识别，需要外部检测器 |
| **CTD / Magi / TextMamba / YSG / CRAFT 等** | ✅ | ❌ | 仅检测，需配 OCR |

---

## 可选检测器：SwinTextSpotter v2 与 DPText-DETR

二者均为可选集成，需要克隆外部仓库，并在参数里设置 `repo_path`。

### SwinTextSpotter v2

- 仓库：<https://github.com/mxin262/SwinTextSpotterv2>
- 在本项目中的检测器名：`swintextspotter_v2`

安装步骤：
1. 克隆并按上游 README 安装依赖。
2. 下载权重（Model Zoo / README）。
3. 在本项目中选择 `swintextspotter_v2` 并设置 `repo_path`。

说明：当前通过子进程调用其 demo 脚本。若上游 CLI 或输出格式变更，需同步调整：
`modules/textdetector/detector_swintextspotter_v2.py`。

### DPText-DETR

- 仓库：<https://github.com/ymy-k/DPText-DETR>
- 在本项目中的检测器名：`dptext_detr`

安装步骤：
1. 克隆并按上游 README 安装依赖。
2. 下载预训练权重。
3. 在本项目中选择 `dptext_detr` 并设置 `repo_path`。

说明：当前会尝试 `demo.py` / `tools/demo.py` / `eval.py`。如上游接口不同，需调整：
`modules/textdetector/detector_dptext_detr.py`（CLI 参数、输出路径、JSON 结构）。

---

## CRAFT（craft_det）

独立 CRAFT 文本检测（支持弯曲/场景文本），不依赖 EasyOCR。

- 检测器：`craft_det`
- 安装：`pip install craft-text-detector torch`
- 无需 `repo_path`

可搭配 OCR：`surya_ocr` / `paddle_rec_v5` / `manga_ocr`。

注意依赖冲突：`craft-text-detector` 依赖 `opencv-python<4.5.4.62`，而主应用通常使用 `opencv>=4.8`。若冲突，建议改用 `easyocr_det` 或 `mmocr_det`，并参考 `docs/OPTIONAL_DEPENDENCIES.md`。

---

## HF 通用目标检测（hf_object_det）

可加载任意 Hugging Face 目标检测模型（DETR / RT-DETR 等），适合在 HF 上已有漫画文本检测模型时使用。

- 检测器：`hf_object_det`
- 安装：`pip install transformers torch`
- 关键参数：
  - `model_id`（例如 `facebook/detr-resnet-50`）
  - `score_threshold`
  - `labels_include`（可选，逗号分隔）


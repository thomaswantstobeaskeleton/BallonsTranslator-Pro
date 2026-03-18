# Machine translation models (Chinese → English) and subtitle layout

## Sample: how subtitle lines read as one paragraph

From a typical video run, the **translated lines** (one per on-screen subtitle) in order look like this when read as a single paragraph:

**Paragraph (lines in order):**

> Filing Number: 1904073220604077. This seems to be the bus that took me to Chu Province for university. Wasn't I undergoing the Heavenly Tribulation? Could it be that… I'm back. I'm Chen Beixuan, around thirty years old, taken away from Earth by the Cangqin. From then on I set out on the path of cultivation. Because of his astonishing talent—

So each line is **one subtitle** (one bubble or beat). The flow fixer (and you) can judge whether this reads smoothly when concatenated; the flow fixer may change a line (e.g. "I" → "I'm Chen Beixuan", or "his" → "my") so the sequence reads better. Green text in the app marks lines the flow fixer revised.

---

## Best models for Chinese → English MT (research summary)

Benchmarks used in the community:

- **WMT (Conference on Machine Translation)** — shared tasks each year; WMT25 includes Chinese–English. Systems are ranked by automatic metrics (e.g. COMET, BLEU) and sometimes human evaluation.
- **FLORES-200** — many language pairs; Chinese–English is a main direction. Often reported with **XCOMET** (or similar) scores.
- **Mandarin / minority-language** — some reports also test Mandarin and Chinese minority languages.

### Top open-weight MT models (Chinese–English, 2024–2025)

| Model | Size | Notes | VRAM (approx) |
|-------|------|--------|----------------|
| **HY-MT1.5-7B** (Tencent) | 7B | WMT25-style champion; 33 languages, terminology/context/formatted translation. Best quality in its class. | ~14 GB FP16; **~4–5 GB Q4** |
| **HY-MT1.5-1.8B** (Tencent) | 1.8B | Lighter; “comparable to 7B” per Tencent; FP8/Int4 for edge. | ~2–3 GB; ~1–2 GB quantized |
| **Marco-MT** (Alibaba, Qwen3-14B) | 14B | WMT2025 top En↔Zh; not 7B-class. | Much higher VRAM |
| **LMT** (NiuTrans) | 0.6B–8B | Multilingual; Zh–En is a focus. | Scale with size |

So for **machine translation specifically** (not general chat), **HY-MT 1.5** is the main open option in the 2B–7B range, with the **7B** version giving the best quality and the **1.8B** version for low VRAM.

---

## Should you use HY-MT 1.5 7B quantized (Q4 or less)?

**Short answer: yes, Q4 is a good trade-off for translation.**

- **Translation is not multi-step reasoning.** It’s one-shot: source in → translation out. So the “reasoning degrades more than knowledge under quantization” result applies less. Lexical and grammatical knowledge (and the MT training) tend to survive Q4 well.
- **Official options for HY-MT1.5-7B:**  
  - **GPTQ-Int4** (Hugging Face): [tencent/HY-MT1.5-7B-GPTQ-Int4](https://huggingface.co/tencent/HY-MT1.5-7B-GPTQ-Int4)  
  - **GGUF Q4_K_M** (e.g. for llama.cpp/LM Studio): [tencent/HY-MT1.5-7B-GGUF](https://huggingface.co/tencent/HY-MT1.5-7B-GGUF) — Q4_K_M ≈ 4.6 GB, Q6_K ≈ 6.2 GB, Q8_0 ≈ 8 GB.
- **FP8** is also offered by Tencent if your stack supports it; quality is between full precision and Q4.

So: **use HY-MT 1.5 7B with Q4_K_M (or GPTQ-Int4)** if you can afford ~4–5 GB VRAM. You get a dedicated Chinese–English (and 33-language) MT model with much better quality than the 2B version, and Q4 keeps “knowledge” (translation quality) largely intact while cutting VRAM. Going to Q3 or lower is possible but may start to show more quality loss; Q4 is the usual recommendation.

---

## Summary

- **Layout:** One line in the app = one subtitle; the paragraph above shows how those lines read in sequence.
- **MT models:** For Chinese → English, **HY-MT1.5-7B** (and its 1.8B variant) are the leading open MT-specific models; benchmarks are WMT and FLORES (e.g. XCOMET).
- **Quantization:** **Q4 (e.g. Q4_K_M or GPTQ-Int4)** on HY-MT1.5-7B is a good choice: translation quality (knowledge) stays strong while VRAM drops to ~4–5 GB. Use the 7B Q4 model over the 2B if your GPU allows.

# Flow fixer setup: local models and ChatGPT

The **flow fixer** runs a second pass after translation to smooth subtitle flow (continuity, pronouns, natural phrasing). You can use a **local model** (free, no rate limits) or **OpenAI/ChatGPT** (uses your credits).

## How far back do revisions apply?

- The fixer receives the **last N previous subtitle entries** (N = **Context lines**, default **20**, configurable **1–50** in Video translator → Flow fixer → **Context lines**).
- From those entries it flattens translation lines and sends all of them to the model (plus the new line(s) to add). So you can use **much more previous context** (e.g. 20–50 lines) for better flow.
- **When context is long** (e.g. >20 lines and near token limit), the flow fixer model is first asked to **summarize** the older portion into a few short lines; then the main flow pass uses **summary + most recent lines**, so you stay within context limits while keeping recent dialogue intact.
- **Parts:** Previous lines are shown to the model in **parts** (every 5 lines = one part). That helps the model treat segments that are apart from each other as distinct and only suggest revisions where flow is actually broken.
- When the model returns **revised_previous**, those revisions are applied to soft subtitles (SRT/ASS/VTT), the in-memory list, and the translator cache for future hardcoded text. If the model returns a different number of lines, the code pads or trims so revisions still apply consistently.

---

## Option 1: OpenAI / ChatGPT (use your $5 credits)

**Best for:** Quick setup, no local GPU; good quality with cheap models.

### Setup

1. Get an API key: [platform.openai.com](https://platform.openai.com) → **API keys** → Create key. Add payment if needed (e.g. $5 credits).
2. In **Video translator** → **Flow fixer**:
   - Enable **Use flow fixer**.
   - Choose **OpenAI / ChatGPT (use credits)**.
   - Paste your **OpenAI API key** (starts with `sk-`).
   - Set **OpenAI model** to one of:
     - **`gpt-4o-mini`** (default) — cheap, fast, good for flow. ~$0.15/1M input, ~$0.60/1M output.
     - **`gpt-3.5-turbo`** — even cheaper; flow quality still fine for subtitles.

3. Run the video pipeline. Translation can stay on OpenRouter (or anything); flow fixer uses only OpenAI.

**Rough cost:** Flow fixer does one short request per subtitle batch. A long video might use a few hundred requests; with gpt-4o-mini, $5 can cover many hours of flow fixing.

---

## Option 2: Local model (Ollama or LM Studio)

**Best for:** No API cost, no rate limits; you need a GPU or strong CPU.

### A) Ollama (simplest)

1. Install [Ollama](https://ollama.com).
2. Pull a small model (one is enough for flow):
   ```bash
   ollama pull qwen2.5:3b
   ```
   Or: `phi3:mini`, `llama3.2:3b`, `gemma2:2b`.
3. Start the server (Ollama usually runs in the background after install).
4. In **Video translator** → **Flow fixer**:
   - Enable **Use flow fixer**.
   - Choose **Local server (Ollama or LM Studio)**.
   - **Server URL:** `http://localhost:11434/v1`
   - **Model (local):** the Ollama model name, e.g. `qwen2.5:3b`, `phi3:mini`, `llama3.2:3b`.

### B) LM Studio

1. Install [LM Studio](https://lmstudio.ai).
2. Download a small model (e.g. **Qwen 2.5 3B**, **Phi-3 mini**, **Llama 3.2 3B**).
3. In LM Studio, **Start Server** (often port 1234).
4. In **Video translator** → **Flow fixer**:
   - Enable **Use flow fixer**.
   - Choose **Local server (Ollama or LM Studio)**.
   - **Server URL:** `http://localhost:1234/v1`
   - **Model (local):** the name shown in LM Studio when the server is running (often `local` or the model filename).

**Structured output (recommended):** If LM Studio (or your local server) supports **structured output**, the **app sends the correct JSON Schema automatically** with each request (with the right counts for that request). You usually do **not** need to paste a schema in LM Studio—leave the schema field empty so the request from the app is used.

If LM Studio requires you to paste a schema in the UI (and it must be valid JSON), use the one below. Replace the numbers with your usual **Context lines** (e.g. 20) and 1 for new lines. This is only a fallback; the app still sends the exact schema per request when possible.

```json
{
  "type": "object",
  "properties": {
    "revised_previous": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 20,
      "maxItems": 20,
      "description": "One string per previous line, same order."
    },
    "revised_new": {
      "type": "array",
      "items": {"type": "string"},
      "minItems": 1,
      "maxItems": 1,
      "description": "One string per new line, same order."
    }
  },
  "required": ["revised_previous", "revised_new"],
  "additionalProperties": false
}
```

**Note:** The code uses variables `n_prev` and `n_new`; in JSON you must use actual numbers (e.g. 20 and 1). If you use a pasted schema, set the numbers to match your flow fixer **Context lines** and typical new-line count (usually 1).

**LM Studio settings that affect the flow fixer (avoid these causing errors):**

- **Context length** — Must be large enough for the prompt (system + user with 10–20+ subtitle lines). If the model’s context is too small, output can be cut off and you get invalid JSON. Use at least **4096** (or 8192) in the model / server settings.
- **Max new tokens** — The app sends `max_tokens` (default **512**) in the API request. If LM Studio caps responses below that (e.g. 256), the JSON can be truncated and parsing fails. Set LM Studio’s “Max new tokens” (or equivalent) to **at least 512** so the server can return full responses, or rely on the app’s request (if your LM Studio version honors the API’s `max_tokens`).
- **Stop sequences** — The app sends `stop=["</think>"]` so the model stops before emitting reasoning. If your model is “think”-style and you have **</think>** in LM Studio’s stop list, that’s good. If the server ignores the API’s stop and the model outputs long `</think>...` before JSON, you’ll see “invalid JSON” or “model returned only reasoning”; use a model that doesn’t lead with reasoning, or ensure the server respects the stop sequence from the request.
- **Temperature** — The app uses low temperature (0.1, and 0 on retry) for stable JSON. If LM Studio overrides with a higher temperature in the UI, output can be less consistent. Prefer letting the API request control temperature, or set a low value (e.g. 0.1–0.2) in LM Studio when using the flow fixer.
- **Instruction-following models** — The flow fixer needs **only a JSON object** (no markdown, no `</think>`, no explanation). Models that always wrap output in markdown or emit long reasoning often cause parse errors. Use a model that follows “reply with only JSON” well (e.g. Qwen 2.5, Phi-3, Llama 3.2 instruction-tuned).

If you see **“revised_previous length X (expected Y)”**, **“invalid JSON”**, or **“model returned only reasoning”**, check the points above; the app will still pad/trim and retry where it can, but correct LM Studio (or local server) setup reduces these errors.

---

## Recommended local models (flow-only task)

| Model            | Size  | VRAM (approx) | Notes                    |
|------------------|-------|----------------|--------------------------|
| **Qwen2.5:3b**   | 3B    | ~2–3 GB        | Good balance (Ollama).   |
| **Phi-3 mini**   | 3.8B  | ~2–4 GB        | Fast, good instructions.|
| **Llama 3.2 3B** | 3B    | ~2–3 GB        | Solid default.           |
| **Gemma 2 2B**   | 2B    | ~1.5–2 GB      | Smallest; still usable.  |

Flow fixing is light: the model only rewrites a few lines for continuity. A 2B–3B model is enough; larger models are optional.

---

## LM Studio / local models for main translation (low VRAM)

If you use **LM Studio** (or Ollama) for the **main translator** (not just the flow fixer), small 2B **translation-only** models (e.g. HY MT 1.5 2B) often struggle with:

- **Instruction-following** (JSON output, glossary, “use this term”)
- **Consistent terminology** and flow

**Better low-VRAM options** (instruction-tuned, same ~2–4 GB VRAM):

| Model (LM Studio / Ollama) | Size  | VRAM (approx) | Notes                          |
|----------------------------|-------|----------------|--------------------------------|
| **Qwen2.5 3B Instruct**   | 3B    | ~2–3 GB        | Strong instructions, good CN→EN. |
| **Phi-3 mini**             | 3.8B  | ~2–4 GB        | Fast, follows prompts well.    |
| **Llama 3.2 3B Instruct**  | 3B    | ~2–3 GB        | Reliable default.              |
| **Qwen2.5 1.5B Instruct** | 1.5B  | ~1–2 GB        | Smallest; better than 2B MT-only. |
| **Gemma 2 2B**             | 2B    | ~1.5–2 GB      | OK; 3B usually better.         |

Use the **exact model name** shown in LM Studio in **Translator** options (and in **Flow fixer → Model (local)** if you use a local flow fixer). For glossary and JSON, 3B instruction models usually beat 2B MT-only models without needing more VRAM.

**Dedicated MT models (Chinese → English):** For best quality at ~4–5 GB VRAM, use **HY-MT1.5-7B** quantized to **Q4** (e.g. GGUF Q4_K_M or GPTQ-Int4). It is built for machine translation and tops WMT/FLORES-style benchmarks; Q4 keeps translation quality close to full precision. See **docs/MT_MODELS_AND_LAYOUT.md** for a sample paragraph of subtitle lines, benchmarks, and quantization notes.

---

## Summary

- **ChatGPT credits:** Flow fixer → **OpenAI / ChatGPT** → paste API key → model **gpt-4o-mini** (or gpt-3.5-turbo). No OpenRouter; no extra rate limits from flow.
- **Local, free:** Flow fixer → **Local server** → Ollama (`http://localhost:11434/v1`, model e.g. `qwen2.5:3b`) or LM Studio (`http://localhost:1234/v1`, model e.g. `local`).

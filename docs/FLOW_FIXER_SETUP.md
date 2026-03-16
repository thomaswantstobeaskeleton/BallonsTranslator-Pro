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

## Summary

- **ChatGPT credits:** Flow fixer → **OpenAI / ChatGPT** → paste API key → model **gpt-4o-mini** (or gpt-3.5-turbo). No OpenRouter; no extra rate limits from flow.
- **Local, free:** Flow fixer → **Local server** → Ollama (`http://localhost:11434/v1`, model e.g. `qwen2.5:3b`) or LM Studio (`http://localhost:1234/v1`, model e.g. `local`).

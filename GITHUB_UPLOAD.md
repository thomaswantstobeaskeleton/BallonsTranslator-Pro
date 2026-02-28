# Uploading this project to GitHub – data protection checklist

Before you push this repo to GitHub (or any public remote), confirm the following so **API keys, passwords, and local data stay private**.

---

## 1. What is already protected (do not remove)

| Item | Status |
|------|--------|
| **config/config.json** | In `.gitignore` – stores all module params (API keys, Stariver User/Password, Google api_key, etc.). **Never commit this file.** |
| **config/config.local.json** | In `.gitignore` – use for local overrides if needed. |
| **.env, .env.*** | In `.gitignore` – use for environment-based secrets if you add support. |
| **data/models** | In `.gitignore` – downloaded model weights. |
| **data/testpacks**, **data/*.png** | In `.gitignore` – test images and packs. |
| **icons**, **release**, **libs**, **venv** | In `.gitignore` – build artifacts and venv. |

No **API keys or passwords** are hardcoded in source. The **Google** translator and **Google Lens** OCR read `api_key` from params (saved in `config.json`). Stariver User/Password are only in `config.json`.

---

## 2. Before your first push – quick checks

1. **Confirm config is ignored**
   ```bash
   git check-ignore config/config.json
   ```
   Should print `config/config.json`. If it doesn’t, ensure `.gitignore` contains `config/config.json`.

2. **Ensure config.json is not staged**
   ```bash
   git status
   ```
   `config/config.json` must **not** appear under “Changes to be committed” or “Untracked files” (if it exists locally, it should be ignored).

3. **Search for accidental secrets in tracked files**
   ```bash
   git grep -i "api_key\|password\|secret" -- "*.py" "*.json" ":!*.md"
   ```
   Review the result. There should be no real keys or passwords; only param names, empty defaults, or placeholder strings like `"填入你的用户名"`.

4. **Optional: list files that will be committed**
   ```bash
   git add -n . 2>/dev/null; git status --short
   ```
   Verify no `config.json`, `.env`, or `data/models` paths are listed.

---

## 3. After cloning (for you or others)

- **config.json** is not in the repo. On first run, the app will create it when you save settings. Alternatively, copy `config/config.example.json` to `config/config.json` and then add your API keys and passwords **only in the app’s settings panel** (they will be stored in `config.json`, which remains gitignored).
- Add your **API keys and passwords only in**:
  - **config/config.json** (via the app’s settings panel), or
  - A local file that is in `.gitignore` (e.g. `config/config.local.json` if you load it yourself), or
  - Environment variables if you later add that support.

---

## 4. If you already committed secrets by mistake

If `config.json` or any file with real keys was committed earlier:

1. **Remove the file from Git history** (e.g. `git filter-branch` or BFG Repo-Cleaner) and force-push, **or**
2. **Rotate all exposed keys/passwords immediately** (Google, Stariver, DeepL, etc.) and remove the file from the repo so future commits don’t contain them.

Prefer rotating keys if the repo was ever public.

---

## 5. “Stable release” folder

There is no separate “stable release” folder. The **whole repo** is safe to upload as long as:

- You don’t add `config/config.json` or other secret files to Git, and  
- You’ve run the checks in **Section 2** before pushing.

For **releases**, use GitHub’s **Releases** (tags + optional source zip). Do not put API keys or `config.json` in release assets.

---

**Summary:** `config.json` and other sensitive paths are gitignored. No API keys are stored in source. Run the checks in Section 2 before pushing, and you can upload this project to GitHub safely.

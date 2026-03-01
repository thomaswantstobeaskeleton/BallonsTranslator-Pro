# How to Propose Changes to BallonsTranslator (Upstream)

Step-by-step instructions based on feedback from upstream maintainer **bropines**. Covers: (1) getting your fork mentioned in the upstream README, (2) cleaning up your fork before the PR, and (3) proposing your fork as an experimental branch (upstream currently has only `dev` and `main`).

---

## 1. Get your fork mentioned in the upstream README

**Goal:** One clear mention in [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) README that points to your fork and explains why it's useful.

### What to do

1. **Open an issue** on the upstream repo: https://github.com/dmMaze/BallonsTranslator/issues (or Discussions if they use it).
2. **Use the copy-paste text below** (title + body). Leave exact placement and wording to them.
3. **Ask once.** Don't repeat the request in every thread.

---

### Copy-paste: Issue (ask to add fork to README)

**Title:**
```
Add community fork (BallonsTranslator-Pro) to README
```

**Body:**
```
Could you add a short note in the README mentioning this community fork? bropines previously suggested something along these lines.

Suggested wording (you can edit as you like):

---

**Community fork with extended features:** There's a fork with more advanced features here: [BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro). I added configurable cropping/box padding for most text detection models (4–6 px works well), a bunch of extra detectors and OCR engines, translation context & glossary, Manga/Comic source (MangaDex, GOMANGA, etc.), batch PDF export, and some other quality-of-life stuff. Default behavior is unchanged unless you turn the new options on.

---

Happy to adjust the wording or provide a shorter one-liner. Thanks for maintaining BallonsTranslator.
```

---

### If using the "Feature Request" template

Upstream uses a **Feature Request** template with separate fields. Fill it like this:

| Field | What to use |
|-------|-------------|
| **Type of Request** | Leave **New Feature** (or choose the option that fits a docs/README change). |
| **Title** | `Add community fork (BallonsTranslator-Pro) to README` — or the template may add a "Feature Request: " prefix; either is fine. |
| **Description** | Paste the full **Body** text from the copy-paste above (the README request and suggested wording). |
| **Version Info** | For this request it doesn't apply. You can put: `N/A — repository/docs request (adding fork mention to README), not an application feature.` Or copy your fork's version from the app console if the field is required. |
| **Pictures** | Leave empty (no screenshot needed). |
| **Additional Information** | Optional. You can leave "Leave a comment" or add: `Happy to adjust the wording or provide a shorter one-liner. Thanks for maintaining BallonsTranslator.` |

Then click **Create**.

Upstream asked for a more readable README and clearer Git history (focused commits, not many mixed edits).

### README

- This repo's README is already broken down point-by-point (**Key highlights**, **Tutorials (step-by-step)**). No change required unless you want to shorten or reorganize further.

### Git history (optional but recommended)

If your branch has many messy commits, prepare a **clean branch** for the PR:

1. **Add upstream as a remote** (if not already):
   ```bash
   git remote add upstream https://github.com/dmMaze/BallonsTranslator.git
   git fetch upstream
   ```

2. **Create a branch from upstream main** for the PR:
   ```bash
   git checkout -b pr-experimental upstream/main
   ```

3. **Merge your fork's changes** (replace `main` with your branch name if different):
   ```bash
   git merge main --no-ff -m "Merge BallonsTranslator-Pro extended features (for experimental branch)"
   ```
   Or, if you prefer a single squashed commit:
   ```bash
   git merge main --squash
   git commit -m "Add BallonsTranslator-Pro features: configurable detection cropping, extra detectors/OCR/inpainters, translation context, Manga source, batch PDF, etc."
   ```

4. **Push the branch** to your fork:
   ```bash
   git push origin pr-experimental
   ```

5. When opening the PR, set **base** to `main` (or `dev`) on `dmMaze/BallonsTranslator` and **compare** to `pr-experimental` from your fork. In the PR description, ask them to create an `experimental` branch and merge there (see copy-paste below).

---

## 3. Propose your fork as an experimental branch

**Goal:** Your changes merged into upstream as a **separate experimental branch**. Upstream currently has only **`dev`** and **`main`**, so the PR will target **`main`** (or `dev`) and the description will ask them to **create** an **`experimental`** branch and merge the PR into it.

### Steps

1. Clean up your fork (see §2 above); use a clean branch if you prepared one.
2. Go to: https://github.com/dmMaze/BallonsTranslator/compare
3. **Base repository:** `dmMaze/BallonsTranslator` — **base branch:** `main` (or `dev` if they prefer).
4. **Head repository:** `thomaswantstobeaskeleton/BallonsTranslator-Pro` — **compare branch:** your PR branch (e.g. `main` or `pr-experimental`).
5. Click **Create pull request** and paste the description below.

---

### Copy-paste: Pull request (experimental branch)

**Title:**
```
Propose BallonsTranslator-Pro as experimental branch
```

**Body:**
```
This PR proposes merging the [BallonsTranslator-Pro](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro) fork into this repository as a **separate experimental branch**, not into `main`, so users can opt in and stable behavior stays unchanged.

**Request:** Since the repo currently has only `dev` and `main`, could you:
1. Create a new branch (e.g. `experimental` or `pro-features`) from `main`, and
2. Merge this PR into that branch (or change the base of this PR to that branch once it exists),

instead of merging into `main` or `dev`?

---

**What this adds (summary):**
- **Configurable cropping/box padding** for most text detection models (4–6 px recommended) to reduce clipped text.
- Additional optional modules: more detectors (e.g. Paddle v5, Surya, HF object-detection, MMOCR), OCR engines, inpainters.
- **Translation context & glossary** (project/series), optional context summarization for LLM translator.
- **Manga/Comic source** (MangaDex incl. raw/original language, GOMANGA, Manhwa Reader, Comick, local folder), **batch export to PDF**, duplicate/overlapping block check, 370+ fonts, and other UI/config improvements.

Original behavior and defaults are unchanged unless the user enables new options. Only code paths related to these features were changed; no unrelated refactors. Happy to fix any regressions or adjust scope if you prefer a smaller first merge.

Full feature list and docs: [BallonsTranslator-Pro README](https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro#readme).
```

---

## 4. Quick reference

| Goal | Action |
|------|--------|
| README mention | Open one issue with the copy-paste in §1. Don't spam. |
| Clean branch for PR | Optional: branch from `upstream/main`, merge or squash your fork, push to `pr-experimental`. |
| Experimental branch PR | Open PR to `main` (or `dev`); use the copy-paste in §3 and ask them to create `experimental` and merge there. |
| Helping users | When someone has a problem your fork fixes: "I solved this like this" + link to the specific code block. See COMMUNITY_RESPONSES.md. |

---

## 5. Links

- **Upstream:** https://github.com/dmMaze/BallonsTranslator  
- **Your fork:** https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro  
- **Commit/style guidelines:** [CONTRIBUTING.md](CONTRIBUTING.md)  
- **Community reply templates:** [COMMUNITY_RESPONSES.md](COMMUNITY_RESPONSES.md)

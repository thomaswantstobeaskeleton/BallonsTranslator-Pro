# Connect to GitHub and (optional) auto-push

## 1. Connect to **your** repository

Right now the repo is wired to the **original** project:

- **origin** → `https://github.com/dmMaze/BallonsTranslator.git`

To push **your** changes you need a repo under your account.

### Option A: Push to your own new repo

1. On GitHub: **New repository** (e.g. `BallonsTranslator` or `BallonsTranslator-fork`). Do **not** add a README or .gitignore if the project already has them.
2. In your project folder, either **replace** the existing remote or **add** a second one:

   **Replace origin with your repo (recommended if this is only your fork):**
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```

   **Or add a second remote (e.g. "mygithub") and push there:**
   ```bash
   git remote add mygithub https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u mygithub dev
   ```

3. Push your branch (e.g. `dev` or `main`):
   ```bash
   git push -u origin dev
   ```
   Use your branch name instead of `dev` if different.

### Option B: Fork on GitHub, then use your fork

1. On GitHub, open https://github.com/dmMaze/BallonsTranslator and click **Fork**.
2. Set your fork as `origin` (or add it as a remote):
   ```bash
   git remote set-url origin https://github.com/YOUR_USERNAME/BallonsTranslator.git
   ```
3. Push:
   ```bash
   git push -u origin dev
   ```

---

## 2. “Automatically upload when we do changes”

Git does **not** push by itself. Normal flow:

1. You change files.
2. You **commit** (e.g. in Cursor: Source Control → stage → “Commit”).
3. You **push** (e.g. “Sync” or “Push” in Cursor).

So “automatically upload” = **automatically push after each commit**.

### Enable auto-push after every commit (optional)

A **post-commit hook** can run `git push` after each successful commit.

**One-time setup (PowerShell, in project root):**

```powershell
# Create post-commit hook that pushes to origin after each commit
$hookPath = ".git/hooks/post-commit"
$content = @"
#!/bin/sh
# Auto-push to origin after commit (branch is already set by first push)
git push origin
"@
[System.IO.File]::WriteAllText((Resolve-Path $hookPath -ErrorAction SilentlyContinue) ?? (Join-Path (Get-Location) ".git/hooks/post-commit"), $content)
# Make executable (Git Bash on Windows)
git update-index --chmod=+x .git/hooks/post-commit 2>$null; if (Test-Path ".git/hooks/post-commit") { Write-Host "Hook created: .git/hooks/post-commit" }
```

Or create the file by hand:

1. Create file **`.git/hooks/post-commit`** (no extension).
2. Put in it:
   ```sh
   #!/bin/sh
   git push origin
   ```
3. Make it executable (in Git Bash):  
   `chmod +x .git/hooks/post-commit`

After that, every time you **commit**, Git will run **push to origin** right after.

**Caveats:**

- Every commit goes to GitHub immediately (no “local-only” commits unless you skip or disable the hook).
- Push can fail (e.g. no network, or branch protection); the commit still stays local.

To **stop** auto-push, delete or rename `.git/hooks/post-commit`.

---

## 3. Summary

| Goal | What to do |
|------|------------|
| Use your own GitHub repo | Create repo (or fork), set `origin` to its URL, then `git push -u origin <branch>`. |
| Upload after each change | After each **commit**, run **Push** (or use the post-commit hook above to do it automatically). |

Once `origin` points to your repo and you’ve pushed once, “Sync” or “Push” in Cursor will upload your latest commits.

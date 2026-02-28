# CUHK Manga Inpainting (cuhk_manga_inpaint)

Seamless Manga Inpainting with Semantics Awareness (SIGGRAPH 2021). Best on high-quality manga when a line map is available.

## Setup

1. **Clone the MangaInpainting repo**
   ```bash
   git clone https://github.com/msxie92/MangaInpainting.git
   cd MangaInpainting
   ```

2. **Install dependencies** (Python 3.6+, PyTorch; see repo README)
   ```bash
   pip install -r requirements.txt
   ```

3. **Download checkpoints** from the [MangaInpainting README](https://github.com/msxie92/MangaInpainting#getting-started):
   - [MangaInpainting](https://drive.google.com/file/d/1YeVwaNfchLhy3lAA7jOLBP-W23onjy8S/view?usp=sharing)
   - [ScreenVAE](https://drive.google.com/file/d/1QaXqR4KWl_lxntSy32QpQpXb-1-EP7_L/view?usp=sharing)

   Place them under `MangaInpainting/checkpoints/` (e.g. `checkpoints/mangainpaintor` and ScreenVAE as required by the repo).

4. **In BallonsTranslator**
   - Choose inpainter **cuhk_manga_inpaint**.
   - Set **repo_path** to the full path of the MangaInpainting repo (the folder that contains `test.py` and `src/`).
   - Set **checkpoints_path** to the checkpoints folder (e.g. `repo_path/checkpoints/mangainpaintor`), or leave empty to use the default.

## Line map

The inpainter generates a **line map** automatically from the image (simple Canny-based extraction). For best results on high-quality manga, you can use an external line extractor (e.g. [MangaLineExtraction](https://github.com/ljsabc/MangaLineExtraction)) and then run the official MangaInpainting script manually; the built-in option is for convenience without extra tools.

## Notes

- First run may be slow (model load). Subprocess timeout is 300 seconds.
- The repo uses `scipy.misc` (deprecated); if you see errors, try a compatible Python/NumPy/SciPy stack or use IOPaint with `--model manga` as an alternative.

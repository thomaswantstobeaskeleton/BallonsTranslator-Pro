# BallonsTranslator-Pro
[简体中文](/README.md) | [English](/README.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | [日本語](../doc/README_JA.md) | [Indonesia](../doc/README_ID.md) | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md) | [Magyar](../doc/README_HU.md)

BallonsTranslator-Pro adalah fork lanjutan dari [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) untuk alur kerja terjemahan manga/komik yang serius.

## Ringkasan alur

1. Deteksi balon/area teks
2. OCR teks
3. Terjemahkan teks
4. Hapus teks asli dari artwork (inpaint)
5. Edit dan ekspor halaman

- Riwayat perubahan: [docs/CHANGELOG.md](../docs/CHANGELOG.md)

## Mulai cepat

- **Windows (disarankan):** jalankan `launcher.bat`.
- **Windows cepat:** jalankan `launch_win.bat`.
- **Lintas platform:** `python launch.py`.
- Panduan GPU: [docs/GPU_ACCELERATION.md](../docs/GPU_ACCELERATION.md).

## Instal Google Fonts (di aplikasi)

1. **Tools → Models → Install Google Font...**
2. Masukkan nama font (mis. `Bangers`, `Noto Sans JP`).
3. Font akan diunduh dan didaftarkan otomatis ke `fonts/google/`.

## Cara clone / download

- ZIP dari GitHub (**Code → Download ZIP**), lalu jalankan app.
- Atau:

```bash
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
cd BallonsTranslator-Pro
python launch.py
```

## Persyaratan

- Python 3.10.2+
- Internet untuk setup/model pertama
- Ruang disk yang cukup untuk model

## Dokumen penting

- [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
- [docs/QUALITY_RANKINGS.md](../docs/QUALITY_RANKINGS.md)
- [docs/MODELS_REFERENCE.md](../docs/MODELS_REFERENCE.md)
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](../docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- [docs/INDESIGN_LPTXT_WORKFLOW.md](../docs/INDESIGN_LPTXT_WORKFLOW.md)

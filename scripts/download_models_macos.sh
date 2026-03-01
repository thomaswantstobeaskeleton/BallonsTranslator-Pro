#!/usr/bin/env bash
# Model download script for macOS (#126).
# Uses curl (default on macOS). Run from repo root or scripts/:
#   ./scripts/download_models_macos.sh
# Or: bash scripts/download_models_macos.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
ROOT_DIR="$(cd ../.. && pwd)"
MODELS_DIR="$ROOT_DIR/data/models"
LIBS_DIR="$ROOT_DIR/data/libs"

echo "Project root: $ROOT_DIR"
echo "Models dir:   $MODELS_DIR"
echo "Libs dir:     $LIBS_DIR"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

download() {
    local url="$1"
    local out="${2:-$(basename "$url")}"
    if [ -f "$out" ]; then
        echo "Skipping (exists): $out"
        return
    fi
    echo "Downloading: $out"
    curl -L -C - -o "$out" "$url"
}

# Comic Text Detector
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt"
# Comic Text Detector for CPU
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/comictextdetector.pt.onnx"
# AOT Inpainter
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting.ckpt" "aot_inpainter.ckpt"
# LaMa Inpainter
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt" "lama_mpe.ckpt"
# Sugoi Translator
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/sugoi-models.zip"
unzip -o -d sugoi_translator sugoi-models.zip 2>/dev/null || true
# MIT_48PX_CTC OCR
download "https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/ocr-ctc.zip"
unzip -o ocr-ctc.zip
mv -f ocr-ctc.ckpt mit48pxctc_ocr.ckpt 2>/dev/null || true
rm -f alphabet-all-v5.txt 2>/dev/null || true
# Manga OCR
if [ ! -d "manga-ocr-base" ]; then
    git lfs install 2>/dev/null || true
    git clone "https://huggingface.co/kha-white/manga-ocr-base"
fi

mkdir -p "$LIBS_DIR"
cd "$LIBS_DIR"
if [ ! -f "libpatchmatch.so" ]; then
    git clone --depth 1 https://github.com/vacancy/PyPatchMatch PyPatchMatch
    cd PyPatchMatch
    NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
    make -j"${NCPU}"
    mv libpatchmatch.so "$LIBS_DIR"
    cd ..
    rm -rf PyPatchMatch
fi

echo "Done. Models in $MODELS_DIR, libs in $LIBS_DIR"

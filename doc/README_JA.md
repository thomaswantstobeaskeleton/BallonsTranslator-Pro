# BallonsTranslator-Pro
[简体中文](/README.md) | [English](/README.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | [日本語](../doc/README_JA.md) | [Indonesia](../doc/README_ID.md) | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md) | [Magyar](../doc/README_HU.md)

BallonsTranslator-Pro は [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator) の発展フォークで、本格的な漫画翻訳ワークフロー向けです。

## ワークフロー概要

1. 吹き出し/テキスト領域の検出
2. OCR 認識
3. 翻訳
4. 原文の除去（inpaint）
5. 編集と書き出し

- 変更履歴: [docs/CHANGELOG.md](../docs/CHANGELOG.md)

## クイックスタート

- **Windows（推奨）:** `launcher.bat`
- **Windows簡易起動:** `launch_win.bat`
- **クロスプラットフォーム:** `python launch.py`
- GPU: [docs/GPU_ACCELERATION.md](../docs/GPU_ACCELERATION.md)

## Google Fonts のインストール（アプリ内）

1. **Tools → Models → Install Google Font...**
2. フォントファミリー名を入力
3. `fonts/google/` に自動登録

## クローン / ダウンロード

```bash
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
cd BallonsTranslator-Pro
python launch.py
```

## 必要条件

- Python 3.10.2+
- 初回セットアップ/モデル取得用のネット接続
- モデル保存用のディスク容量

## 主要ドキュメント

- [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
- [docs/QUALITY_RANKINGS.md](../docs/QUALITY_RANKINGS.md)
- [docs/MODELS_REFERENCE.md](../docs/MODELS_REFERENCE.md)
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](../docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- [docs/INDESIGN_LPTXT_WORKFLOW.md](../docs/INDESIGN_LPTXT_WORKFLOW.md)

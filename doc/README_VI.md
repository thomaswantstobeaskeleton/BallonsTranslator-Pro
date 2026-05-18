# BallonsTranslator-Pro
[简体中文](/README.md) | [English](/README.md) | [pt-BR](../doc/README_PT-BR.md) | [Русский](../doc/README_RU.md) | [日本語](../doc/README_JA.md) | [Indonesia](../doc/README_ID.md) | [Tiếng Việt](../doc/README_VI.md) | [한국어](../doc/README_KO.md) | [Español](../doc/README_ES.md) | [Français](../doc/README_FR.md) | [Magyar](../doc/README_HU.md)

BallonsTranslator-Pro là nhánh nâng cao của [dmMaze/BallonsTranslator](https://github.com/dmMaze/BallonsTranslator), tập trung vào quy trình dịch manga/comic nghiêm túc.

## Luồng xử lý chính

1. Phát hiện bong bóng/vùng chữ
2. OCR nhận dạng chữ
3. Dịch văn bản
4. Xóa chữ gốc khỏi ảnh (inpaint)
5. Chỉnh sửa và xuất trang

- Lịch sử thay đổi: [docs/CHANGELOG.md](../docs/CHANGELOG.md)

## Khởi chạy nhanh

- **Windows (khuyên dùng):** `launcher.bat`
- **Windows nhanh:** `launch_win.bat`
- **Đa nền tảng:** `python launch.py`
- Hướng dẫn GPU: [docs/GPU_ACCELERATION.md](../docs/GPU_ACCELERATION.md)

## Cài Google Fonts trong app

1. **Tools → Models → Install Google Font...**
2. Nhập tên font (ví dụ `Bangers`, `Noto Sans JP`)
3. Ứng dụng tự tải và đăng ký vào `fonts/google/`

## Clone / tải mã nguồn

```bash
git clone https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro
cd BallonsTranslator-Pro
python launch.py
```

## Yêu cầu

- Python 3.10.2+
- Internet cho lần setup/model đầu tiên
- Đủ dung lượng đĩa cho model

## Tài liệu quan trọng

- [docs/TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
- [docs/QUALITY_RANKINGS.md](../docs/QUALITY_RANKINGS.md)
- [docs/MODELS_REFERENCE.md](../docs/MODELS_REFERENCE.md)
- [docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md](../docs/TRANSLATION_CONTEXT_AND_GLOSSARY.md)
- [docs/INDESIGN_LPTXT_WORKFLOW.md](../docs/INDESIGN_LPTXT_WORKFLOW.md)

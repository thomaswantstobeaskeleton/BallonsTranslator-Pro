def should_enable_auto_textlayout(let_autolayout_flag: bool, let_fntsize_flag: int, enable_detect: bool, enable_ocr: bool, enable_translate: bool) -> bool:
    return (
        bool(let_autolayout_flag)
        and (int(let_fntsize_flag) == 0)
        and (bool(enable_detect) or bool(enable_ocr) or bool(enable_translate))
    )


def is_detect_only_run(enable_detect: bool, enable_ocr: bool, enable_translate: bool) -> bool:
    return bool(enable_detect) and (not bool(enable_ocr)) and (not bool(enable_translate))


def should_run_post_detect_autofit(let_autolayout_flag: bool, let_fntsize_flag: int, enable_detect: bool, enable_ocr: bool, enable_translate: bool) -> bool:
    return should_enable_auto_textlayout(
        let_autolayout_flag,
        let_fntsize_flag,
        enable_detect,
        enable_ocr,
        enable_translate,
    ) and is_detect_only_run(enable_detect, enable_ocr, enable_translate)

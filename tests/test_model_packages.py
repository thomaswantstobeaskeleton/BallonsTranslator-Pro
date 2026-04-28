import types

from utils.model_packages import get_module_classes_for_packages


def _mk_cls(name: str, has_download: bool = True):
    attrs = {"__name__": name}
    attrs["download_file_list"] = ["weights.bin"] if has_download else None
    return types.SimpleNamespace(**attrs)


def test_selected_packages_map_to_expected_module_classes():
    ctd = _mk_cls("CTD")
    aot = _mk_cls("AOT")
    manga_ocr = _mk_cls("MangaOCR")
    paddle_vl = _mk_cls("PaddleOCRVLManga")

    registries = {
        "textdetector": {"ctd": ctd},
        "inpaint": {"aot": aot},
        "ocr": {
            "manga_ocr": manga_ocr,
            "PaddleOCRVLManga": paddle_vl,
            # This entry exists but should be skipped because it has no download files.
            "mit48px_ctc": _mk_cls("MIT48CTC", has_download=False),
        },
        "translator": {},
    }

    selected = ["core", "advanced_ocr"]
    classes = get_module_classes_for_packages(selected, registries=registries)

    assert classes == [ctd, aot, manga_ocr, paddle_vl]


def test_empty_selection_returns_no_modules():
    assert get_module_classes_for_packages([], registries={}) == []


def test_unknown_package_ids_are_ignored():
    ctd = _mk_cls("CTD")
    registries = {
        "textdetector": {"ctd": ctd},
        "inpaint": {},
        "ocr": {},
        "translator": {},
    }

    classes = get_module_classes_for_packages(["core", "does_not_exist"], registries=registries)
    assert classes == [ctd]


def test_no_core_selection_only_uses_selected_non_core_packages():
    paddle_vl = _mk_cls("PaddleOCRVLManga")
    mit48 = _mk_cls("MIT48")
    mit32 = _mk_cls("MIT32")
    registries = {
        "textdetector": {"ctd": _mk_cls("CTD")},
        "inpaint": {"aot": _mk_cls("AOT")},
        "ocr": {
            "PaddleOCRVLManga": paddle_vl,
            "mit48px": mit48,
            "mit32px": mit32,
        },
        "translator": {},
    }

    classes = get_module_classes_for_packages(["advanced_ocr"], registries=registries)

    assert classes == [paddle_vl, mit48, mit32]

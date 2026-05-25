"""Regression tests for upstream PR ports (source-level assertions)."""

# --- Test #1163: bare except fix ---
def test_shared_bare_except():
    with open("utils/shared.py", "r", encoding="utf8") as f:
        source = f.read()
    # The specific cache-loading block should use 'except Exception:'
    assert "except Exception:\n                print(f'cached file" in source


# --- Test #1130: blk.xyxy updated in setRect ---
def test_textitem_setRect_updates_xyxy():
    with open("ui/textitem.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "def _xywh_to_xyxy(r):" in source
    assert "self.blk.xyxy = _xywh_to_xyxy(rect)" in source


# --- Test #1152: MergeBlkItemsCommand exists ---
def test_merge_blk_items_command_exists():
    with open("ui/textedit_commands.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "class MergeBlkItemsCommand(QUndoCommand):" in source
    assert "self.ctrl.deleteTextblkItemList(self.secondary_list" in source
    assert "self.ctrl.recoverTextblkItemList(self.secondary_list" in source


# --- Test #1027: XPU patches structure ---
def test_yolo_make_grid_xpu_guard():
    with open("modules/textdetector/yolov5/yolo.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "d.type == 'xpu'" in source
    assert "torch.meshgrid([torch.arange(ny), torch.arange(nx)]" in source


def test_yolov5_utils_nms_xpu_guard():
    with open("modules/textdetector/yolov5/yolov5_utils.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "x.device.type == 'xpu'" in source
    assert "j.to('cpu').float().to(x.device)" in source


# --- Test label color dialog initial value ---
def test_label_color_dialog_initial():
    with open("ui/custom_widget/label.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "QColorDialog.getColor(initial_color)" in source
    assert "initial_color = self.color if self.color is not None else QColor(255, 255, 255)" in source


# --- Test mainwindow uses undo command for merge ---
def test_mainwindow_merge_uses_command():
    with open("ui/mainwindow.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "MergeBlkItemsCommand(blks, self.st_manager)" in source


# --- Test RealtimeOverlayWidget uses always-on-top frameless flags ---
def test_realtime_overlay_window_flags():
    with open("ui/realtime_overlay.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "WindowStaysOnTopHint" in source
    assert "FramelessWindowHint" in source
    assert "WA_TransparentForMouseEvents" in source
    assert "WA_TranslucentBackground" in source
    assert "HWND_TOPMOST" in source


# --- Test RealtimeTranslatorDialog uses RealtimeOverlayWidget ---
def test_realtime_dialog_uses_overlay_widget():
    with open("ui/realtime_translator_dialog.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "RealtimeOverlayWidget" in source
    assert "module_manager" in source
    assert "_OcrTrWorker" in source
    assert "self._ocr_image" in source
    assert "self._translate_text" in source


# --- Test WindowsNativeBackend is real not stub ---
def test_windows_native_backend_is_real():
    with open("utils/realtime_mode.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "backend_name = \"windows_native\"" in source
    assert "_win32gui.GetDesktopWindow" in source
    assert "PrintWindow" in source
    assert "EnumWindows" in source


# --- Test mainwindow passes module_manager to dialog ---
def test_mainwindow_passes_module_manager():
    with open("ui/mainwindow.py", "r", encoding="utf8") as f:
        source = f.read()
    assert "module_manager=self.module_manager" in source

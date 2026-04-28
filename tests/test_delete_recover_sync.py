import importlib
import sys
import types

import numpy as np


def _install_scenetext_stubs():
    class DummyUndoCommand:
        def __init__(self, parent=None):
            self.parent = parent

    # qtpy stubs
    qtwidgets = types.ModuleType("qtpy.QtWidgets")
    qtwidgets.QApplication = object
    qtwidgets.QWidget = object
    qtwidgets.QGraphicsItem = object
    qtwidgets.QUndoCommand = DummyUndoCommand

    qtcore = types.ModuleType("qtpy.QtCore")
    qtcore.QObject = object
    qtcore.QRectF = object
    qtcore.Qt = object
    qtcore.Signal = lambda *_, **__: None
    qtcore.QPointF = object
    qtcore.QPoint = object

    qtgui = types.ModuleType("qtpy.QtGui")
    qtgui.QKeyEvent = object
    qtgui.QTextCursor = object
    qtgui.QFontMetricsF = object
    qtgui.QFont = object
    qtgui.QTextCharFormat = object
    qtgui.QClipboard = object
    qtgui.QUndoCommand = DummyUndoCommand

    qtpy = types.ModuleType("qtpy")
    sys.modules["qtpy"] = qtpy
    sys.modules["qtpy.QtWidgets"] = qtwidgets
    sys.modules["qtpy.QtCore"] = qtcore
    sys.modules["qtpy.QtGui"] = qtgui

    # ui module stubs required by scenetext_manager imports
    textitem = types.ModuleType("ui.textitem")
    textitem.TextBlkItem = type("TextBlkItem", (), {})
    textitem.TextBlock = type("TextBlock", (), {})
    sys.modules["ui.textitem"] = textitem

    canvas_mod = types.ModuleType("ui.canvas")
    canvas_mod.Canvas = type("Canvas", (), {})
    sys.modules["ui.canvas"] = canvas_mod

    te_area = types.ModuleType("ui.textedit_area")
    for name in [
        "TransTextEdit",
        "SourceTextEdit",
        "TransPairWidget",
        "SelectTextMiniMenu",
        "TextEditListScrollArea",
        "QVBoxLayout",
        "Widget",
    ]:
        setattr(te_area, name, type(name, (), {}))
    sys.modules["ui.textedit_area"] = te_area

    te_cmd = types.ModuleType("ui.textedit_commands")
    te_cmd.propagate_user_edit = lambda *_, **__: None
    for name in [
        "TextEditCommand",
        "ReshapeItemCommand",
        "MoveBlkItemsCommand",
        "AutoLayoutCommand",
        "ApplyFontformatCommand",
        "RotateItemCommand",
        "WarpItemCommand",
        "TextItemEditCommand",
        "PageReplaceOneCommand",
        "PageReplaceAllCommand",
        "MultiPasteCommand",
        "ResetAngleCommand",
        "SqueezeCommand",
    ]:
        setattr(te_cmd, name, type(name, (), {}))
    sys.modules["ui.textedit_commands"] = te_cmd

    text_panel = types.ModuleType("ui.text_panel")
    text_panel.FontFormatPanel = type("FontFormatPanel", (), {})
    sys.modules["ui.text_panel"] = text_panel

    # utils stubs used at import-time
    utils_font = types.ModuleType("utils.fontformat")
    utils_font.FontFormat = type("FontFormat", (), {})
    utils_font.pt2px = lambda v: v
    sys.modules["utils.fontformat"] = utils_font

    cfg = types.ModuleType("utils.config")
    cfg.pcfg = types.SimpleNamespace(module=types.SimpleNamespace(layout_balloon_shape_model_id=""))
    sys.modules["utils.config"] = cfg

    shared = types.ModuleType("utils.shared")
    sys.modules["utils.shared"] = shared

    imgproc = types.ModuleType("utils.imgproc_utils")
    imgproc.extract_ballon_region = lambda *_, **__: None
    imgproc.rotate_polygons = lambda *_, **__: None
    imgproc.get_block_mask = lambda *_, **__: (None, None)
    imgproc.classify_bubble_shape_from_mask = lambda *_: None
    imgproc.mask_centroid_in_crop = lambda *_, **__: None
    sys.modules["utils.imgproc_utils"] = imgproc

    bubble = types.ModuleType("utils.bubble_shape_model")
    bubble.get_bubble_shape_from_model = lambda *_, **__: None
    sys.modules["utils.bubble_shape_model"] = bubble

    box = types.ModuleType("utils.box_size_check_model")
    box.check_box_size_from_model = lambda *_, **__: None
    sys.modules["utils.box_size_check_model"] = box

    textproc = types.ModuleType("utils.text_processing")
    textproc.seg_text = lambda *_, **__: []
    textproc.is_cjk = lambda *_: False
    sys.modules["utils.text_processing"] = textproc

    textlayout = types.ModuleType("utils.text_layout")
    textlayout.layout_text = lambda *_, **__: None
    sys.modules["utils.text_layout"] = textlayout

    linebreak = types.ModuleType("utils.line_breaking")
    linebreak.split_long_token_with_hyphenation = lambda s: s
    sys.modules["utils.line_breaking"] = linebreak

    logger_mod = types.ModuleType("utils.logger")
    logger_mod.logger = types.SimpleNamespace(info=lambda *_, **__: None)
    sys.modules["utils.logger"] = logger_mod

    osb = types.ModuleType("modules.textdetector.outside_text_processor")
    osb.OSB_LABELS = []
    sys.modules["modules.textdetector.outside_text_processor"] = osb


class FakeTextBlkItem:
    def __init__(self, idx, blk):
        self.idx = idx
        self.blk = blk

    def absBoundingRect(self):
        return object()

    def rotation(self):
        return 0


class FakeHighlighter:
    def __init__(self):
        self.matched_map = {}

    def set_current_span(self, *_):
        return None


class FakeSearchWidget:
    def __init__(self):
        self.search_rstedit_list = []
        self.search_counter_list = []
        self.highlighter_list = []
        self.counter_sum = 0
        self.current_edit = None
        self.current_cursor = None
        self.result_pos = -1
        self.update_counter_calls = 0

    def get_result_edit_index(self, edit):
        try:
            return self.search_rstedit_list.index(edit)
        except ValueError:
            return -1

    def setCurrentEditor(self, edit):
        self.current_edit = edit

    def updateCounterText(self):
        self.update_counter_calls += 1


class FakeCanvas:
    def __init__(self, proj, search_widget):
        self.imgtrans_proj = proj
        self.search_widget = search_widget
        self.saved_drawundo_step = 0

    def updateLayers(self):
        return None


class FakeCtrl:
    def __init__(self, canvas, pairwidgets, live_items):
        self.canvas = canvas
        self.pairwidget_list = pairwidgets
        self.live_items = live_items

    def deleteTextblkItemList(self, blk_list, _):
        for blk in blk_list:
            self.live_items.remove(blk)

    def recoverTextblkItemList(self, blk_list, _):
        for blk in sorted(blk_list, key=lambda b: b.idx):
            self.live_items.insert(blk.idx, blk)


def _load_module_and_fixture(mask_plan):
    _install_scenetext_stubs()
    sys.modules.pop("ui.scenetext_manager", None)
    stm = importlib.import_module("ui.scenetext_manager")

    stm.TextBlkItem = FakeTextBlkItem

    mask_iter = iter(mask_plan)
    stm.get_block_mask = lambda *_: next(mask_iter)

    page_name = "p1"
    blk_objs = [types.SimpleNamespace(name=f"blk{i}") for i in range(3)]
    blk_items = [FakeTextBlkItem(i, blk_objs[i]) for i in range(3)]

    pairwidgets = [
        types.SimpleNamespace(e_trans=object(), e_source=object()),
        types.SimpleNamespace(e_trans=object(), e_source=object()),
        types.SimpleNamespace(e_trans=object(), e_source=object()),
    ]

    sw = FakeSearchWidget()
    other_edit = object()
    sw.search_rstedit_list = [
        pairwidgets[1].e_trans,
        pairwidgets[1].e_source,
        pairwidgets[2].e_trans,
        pairwidgets[2].e_source,
        other_edit,
    ]
    sw.search_counter_list = [2, 3, 5, 7, 11]
    sw.highlighter_list = [FakeHighlighter() for _ in range(5)]
    sw.counter_sum = sum(sw.search_counter_list)
    sw.current_edit = pairwidgets[1].e_trans

    proj = types.SimpleNamespace(
        inpainted_array=np.zeros((4, 4), dtype=np.uint8),
        img_array=np.full((4, 4), 9, dtype=np.uint8),
        mask_array=np.array(
            [[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
        ),
        pages={page_name: blk_objs.copy()},
        current_img=page_name,
    )

    canvas = FakeCanvas(proj, sw)
    ctrl = FakeCtrl(canvas, pairwidgets, blk_items.copy())
    return stm, ctrl, blk_items


def test_delete_recover_sync_and_search_restore():
    stm, ctrl, blk_items = _load_module_and_fixture([
        (np.array([[255, 0], [0, 255]], dtype=np.uint8), (0, 0, 2, 2)),
        (None, None),
    ])

    cmd = stm.DeleteBlkItemsCommand([blk_items[2], blk_items[1]], mode=1, ctrl=ctrl)

    assert [b.idx for b in ctrl.live_items] == [0]
    assert [b.name for b in ctrl.canvas.imgtrans_proj.pages["p1"]] == ["blk0"]

    cmd.redo()
    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.inpainted_array[0:2, 0:2], np.array([[9, 0], [0, 9]], dtype=np.uint8))
    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.mask_array[0:2, 0:2], np.array([[0, 0], [0, 0]], dtype=np.uint8))

    cmd.undo()
    assert [b.idx for b in ctrl.live_items] == [0, 1, 2]
    assert [b.name for b in ctrl.canvas.imgtrans_proj.pages["p1"]] == ["blk0", "blk1", "blk2"]
    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.inpainted_array[0:2, 0:2], np.array([[0, 0], [0, 0]], dtype=np.uint8))
    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.mask_array[0:2, 0:2], np.array([[255, 0], [0, 255]], dtype=np.uint8))

    cmd.redo()
    assert [b.idx for b in ctrl.live_items] == [0]
    assert [b.name for b in ctrl.canvas.imgtrans_proj.pages["p1"]] == ["blk0"]
    assert cmd.sw_changed is True
    assert ctrl.canvas.search_widget.counter_sum == cmd.new_counter_sum
    assert ctrl.canvas.search_widget.update_counter_calls >= 1


def test_mode1_absent_inpaint_rect_is_noop():
    stm, ctrl, blk_items = _load_module_and_fixture([(None, None)])

    # isolate to one block so all mode=1 branches run with absent inpaint rect
    ctrl.live_items = [blk_items[0]]
    ctrl.pairwidget_list = [ctrl.pairwidget_list[0]]
    ctrl.canvas.imgtrans_proj.pages = {"p1": [blk_items[0].blk]}

    cmd = stm.DeleteBlkItemsCommand([blk_items[0]], mode=1, ctrl=ctrl)
    before_inpaint = ctrl.canvas.imgtrans_proj.inpainted_array.copy()
    before_mask = ctrl.canvas.imgtrans_proj.mask_array.copy()

    cmd.redo()
    cmd.undo()

    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.inpainted_array, before_inpaint)
    np.testing.assert_array_equal(ctrl.canvas.imgtrans_proj.mask_array, before_mask)

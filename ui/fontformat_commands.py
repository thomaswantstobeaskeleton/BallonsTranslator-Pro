from typing import List, Callable, Dict
import copy

from qtpy.QtGui import QFont
try:
    from qtpy.QtWidgets import QUndoCommand
except:
    from qtpy.QtGui import QUndoCommand

from . import shared_widget as SW
from utils.fontformat import FontFormat, px2pt
from .textitem import TextBlkItem

global_default_set_kwargs = dict(set_selected=False, restore_cursor=False)
local_default_set_kwargs = dict(set_selected=True, restore_cursor=True)



class TextStyleUndoCommand(QUndoCommand):

    def __init__(self, style_func: Callable, params: Dict, redo_values: List, undo_values: List):
        super().__init__()
        self.style_func = style_func
        self.params = params
        self.redo_values = redo_values
        self.undo_values = undo_values

    def redo(self) -> None:
        self.style_func(values=self.redo_values, **self.params)

    def undo(self) -> None:
        self.style_func(values=self.undo_values, **self.params)


def wrap_fntformat_input(values: str, blkitems: List[TextBlkItem], is_global: bool):
    if is_global:
        blkitems = SW.canvas.selected_text_items()
    else:
        if not isinstance(blkitems, List):
            blkitems = [blkitems]
    values = [values] * len(blkitems)
    return blkitems, values

def font_formating(push_undostack: bool = False, is_property = True):

    def func_wrapper(formatting_func):

        def wrapper(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, set_focus: bool = False, *args, **kwargs):
            if is_global and is_property:
                if hasattr(act_ffmt, param_name):
                    act_ffmt[param_name] = values
                else:
                    print(f'undefined param name: {param_name}')

            blkitems, values = wrap_fntformat_input(values, blkitems, is_global)
            if len(blkitems) > 0:
                if is_property:
                    act_ffmt[param_name] = values[0]
                if push_undostack:
                    params = copy.deepcopy(kwargs)
                    params.update({'param_name': param_name, 'act_ffmt': act_ffmt, 'is_global': is_global, 'blkitems': blkitems})
                    undo_values = [getattr(blkitem.fontformat, param_name) for blkitem in blkitems]
                    cmd = TextStyleUndoCommand(formatting_func, params, values, undo_values)
                    SW.canvas.push_undo_command(cmd)
                else:
                    formatting_func(param_name, values, act_ffmt, is_global, blkitems, *args, **kwargs)
            if set_focus:
                if not SW.canvas.hasFocus():
                    SW.canvas.setFocus()
        return wrapper
    
    return func_wrapper

@font_formating()
def ffmt_change_font_family(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontFamily(value, **set_kwargs)

@font_formating()
def ffmt_change_italic(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontItalic(value, **set_kwargs)

@font_formating()
def ffmt_change_underline(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontUnderline(value, **set_kwargs)

@font_formating()
def ffmt_change_strikethrough(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontStrikethrough(value, **set_kwargs)

@font_formating()
def ffmt_change_font_weight(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontWeight(value, **set_kwargs)

@font_formating()
def ffmt_change_bold(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem] = None, **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    values = [QFont.Weight.Bold if value else QFont.Weight.Normal for value in values]
    # ffmt_change_weight('weight', values, act_ffmt, is_global, blkitems, **kwargs)
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontWeight(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_letter_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setLetterSpacing(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_line_spacing(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setLineSpacing(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_vertical(param_name: str, values: bool, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    # set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setVertical(value)


@font_formating(push_undostack=True)
def ffmt_change_writing_mode(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    from utils.text_rendering import resolve_writing_mode, normalize_writing_mode
    for blkitem, value in zip(blkitems, values):
        mode = normalize_writing_mode(value)
        blkitem.fontformat.writing_mode = mode
        text = blkitem.toPlainText() if hasattr(blkitem, "toPlainText") else ""
        rect = blkitem.absBoundingRect(qrect=True) if hasattr(blkitem, "absBoundingRect") else None
        box = (rect.width(), rect.height()) if rect is not None else None
        resolved = resolve_writing_mode(mode, text, box)
        blkitem.setVertical(resolved == "vertical_rl")
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.writing_mode = mode
            blkitem.blk.fontformat.vertical = resolved == "vertical_rl"

@font_formating(push_undostack=True)
def ffmt_change_fit_mode(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    from utils.text_rendering import normalize_fit_mode
    for blkitem, value in zip(blkitems, values):
        mode = normalize_fit_mode(value)
        blkitem.fontformat.fit_mode = mode
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.fit_mode = mode


@font_formating(push_undostack=True)
def ffmt_change_fit_font_size_min(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        v = max(0.0, float(value or 0.0))
        blkitem.fontformat.fit_font_size_min = v
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.fit_font_size_min = v


@font_formating(push_undostack=True)
def ffmt_change_fit_font_size_max(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        v = max(0.0, float(value or 0.0))
        blkitem.fontformat.fit_font_size_max = v
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.fit_font_size_max = v

@font_formating(push_undostack=True)
def ffmt_change_text_padding(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        value = max(0.0, float(value or 0.0))
        blkitem.fontformat.text_padding = value
        blkitem.setPadding(value)
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.text_padding = value

@font_formating(push_undostack=True)
def ffmt_change_fallback_font_chain(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.fallback_font_chain = str(value or "")
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat.fallback_font_chain = blkitem.fontformat.fallback_font_chain

@font_formating(push_undostack=True)
def ffmt_change_manga_preset(param_name: str, values: str, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    from utils.text_rendering import MANGA_PRESETS, normalize_writing_mode, resolve_writing_mode
    for blkitem, value in zip(blkitems, values):
        preset_id = str(value or "")
        preset = MANGA_PRESETS.get(preset_id)
        if not preset:
            continue
        for key, preset_value in preset.items():
            if key == "label":
                continue
            if hasattr(blkitem.fontformat, key):
                setattr(blkitem.fontformat, key, preset_value)
        blkitem.fontformat.manga_preset = preset_id
        # Apply renderer-facing fields immediately.
        mode = normalize_writing_mode(preset.get("writing_mode", "auto"))
        blkitem.fontformat.writing_mode = mode
        rect = blkitem.absBoundingRect(qrect=True) if hasattr(blkitem, "absBoundingRect") else None
        box = (rect.width(), rect.height()) if rect is not None else None
        resolved = resolve_writing_mode(mode, blkitem.toPlainText() if hasattr(blkitem, "toPlainText") else "", box)
        blkitem.setVertical(resolved == "vertical_rl")
        if "stroke_width" in preset:
            blkitem.setStrokeWidth(float(preset["stroke_width"]))
        if "text_padding" in preset:
            blkitem.setPadding(float(preset["text_padding"]))
        if "alignment" in preset:
            blkitem.setAlignment(int(preset["alignment"]))
        if hasattr(blkitem, "set_fontformat"):
            blkitem.set_fontformat(blkitem.fontformat)
        if hasattr(blkitem, "blk") and blkitem.blk is not None:
            blkitem.blk.fontformat = blkitem.fontformat.deepcopy()

@font_formating()
def ffmt_change_frgb(param_name: str, values: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setFontColor(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_srgb(param_name: str, values: tuple, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setStrokeColor(value, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_stroke_width(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setStrokeWidth(value, **set_kwargs)

@font_formating()
def ffmt_change_font_size(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], clip_size=False, **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        if value < 0:
            continue
        value = px2pt(value)
        blkitem.setFontSize(value, clip_size=clip_size, **set_kwargs)

@font_formating(is_property=False)
def ffmt_change_rel_font_size(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], clip_size=False, **kwargs):
    set_kwargs = global_default_set_kwargs if is_global else local_default_set_kwargs
    for blkitem, value in zip(blkitems, values):
        blkitem.setRelFontSize(value, clip_size=clip_size, **set_kwargs)

@font_formating(push_undostack=True)
def ffmt_change_alignment(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    restore_cursor = not is_global
    for blkitem, value in zip(blkitems, values):
        blkitem.setAlignment(value, restore_cursor=restore_cursor)

@font_formating(push_undostack=True)
def ffmt_change_opacity(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setOpacity(value)

@font_formating(push_undostack=True)
def ffmt_change_line_spacing_type(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    restore_cursor = not is_global
    for blkitem, value in zip(blkitems, values):
        blkitem.setLineSpacingType(value, restore_cursor=restore_cursor)


@font_formating(push_undostack=True)
def ffmt_change_shadow_offset(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setBGAttribute(param_name, value)


@font_formating()
def ffmt_change_gradient_enabled(param_name: str, values: float, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.setGradientAttribute(param_name, value)


ffmt_change_shadow_radius = ffmt_change_shadow_offset
ffmt_change_shadow_strength = ffmt_change_shadow_offset
ffmt_change_shadow_color = ffmt_change_shadow_offset

ffmt_change_gradient_start_color = ffmt_change_gradient_enabled
ffmt_change_gradient_end_color = ffmt_change_gradient_enabled
ffmt_change_gradient_angle = ffmt_change_gradient_enabled
ffmt_change_gradient_size = ffmt_change_gradient_enabled
ffmt_change_gradient_type = ffmt_change_gradient_enabled


@font_formating(push_undostack=True)
def ffmt_change_text_on_path(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.text_on_path = int(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_text_on_path_arc_degrees(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.text_on_path_arc_degrees = float(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_warp_style(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.warp_style = int(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_warp_strength(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.warp_strength = float(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_text_box_corner_radius(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.text_box_corner_radius = max(0.0, float(value))
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_blend_mode(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.blend_mode = int(value) if value is not None else 0
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_outline_only(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.outline_only = bool(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_stroke_outline_outside_only(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.stroke_outline_outside_only = bool(value)
        blkitem.repaint_background()
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_auto_fit_font_size(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        blkitem.fontformat.auto_fit_font_size = bool(value)
        if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
            blkitem.blk.fontformat.auto_fit_font_size = bool(value)
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_overlay_opacity(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        v = max(0.0, min(1.0, float(value)))
        blkitem.fontformat.overlay_opacity = v
        if blkitem.blk is not None:
            blkitem.blk.overlay_opacity = v
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_skew_x(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        v = float(value)
        blkitem.fontformat.skew_x = v
        if blkitem.blk is not None:
            blkitem.blk.skew_x = v
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_skew_y(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        v = float(value)
        blkitem.fontformat.skew_y = v
        if blkitem.blk is not None:
            blkitem.blk.skew_y = v
        blkitem.update()


@font_formating(push_undostack=True)
def ffmt_change_fallback_font_chain(param_name: str, values, act_ffmt: FontFormat, is_global: bool, blkitems: List[TextBlkItem], **kwargs):
    for blkitem, value in zip(blkitems, values):
        chain = str(value or '').strip()
        blkitem.fontformat.fallback_font_chain = chain
        if blkitem.blk is not None and hasattr(blkitem.blk, 'fontformat'):
            blkitem.blk.fontformat.fallback_font_chain = chain
        blkitem.set_fontformat(blkitem.fontformat)
        blkitem.update()

"""
State-based Undo Command Pattern.

Replaces fragile action-based undo (e.g. "add stroke", "remove stroke")
with deterministic state snapshots. Each command records the full state
before and after a change; undo simply restores the old state.

Inspired by Comic Translate's immutable state approach.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from qtpy.QtWidgets import QUndoCommand
except ImportError:
    from qtpy.QtGui import QUndoCommand

from utils.textblock import TextBlock
from utils.fontformat import FontFormat


@dataclass
class TextBlockState:
    """
    Immutable snapshot of a TextBlock's user-mutable state.

    This is intentionally flat (no nested objects) so deep-copy is cheap
    and state comparison is trivial.
    """
    xyxy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    lines: List[str] = field(default_factory=list)
    text: List[str] = field(default_factory=list)
    translation: str = ""
    rich_text: str = ""
    angle: int = 0
    src_is_vertical: Optional[bool] = None
    text_mask: Optional[bytes] = None  # serialized as bytes for immutability
    foreground_image_path: Optional[str] = None
    overlay_opacity: float = 1.0
    skew_x: float = 0.0
    skew_y: float = 0.0
    warp_mode: str = "none"
    warp_quad: List[List[float]] = field(default_factory=lambda: [[0, 0], [1, 0], [1, 1], [0, 1]])
    warp_mesh_size: List[int] = field(default_factory=lambda: [2, 2])

    # FontFormat snapshot
    font_family: str = ""
    font_size: float = 24.0
    stroke_width: float = 0.0
    secondary_stroke_width: float = 0.0
    frgb: List[int] = field(default_factory=lambda: [0, 0, 0])
    srgb: List[int] = field(default_factory=lambda: [0, 0, 0])
    secondary_srgb: List[int] = field(default_factory=lambda: [255, 255, 255])
    bold: bool = False
    underline: bool = False
    strikethrough: bool = False
    italic: bool = False
    alignment: int = 0
    vertical: bool = False
    writing_mode: str = "auto"
    fit_mode: str = "shrink"
    fit_font_size_min: float = 0.0
    fit_font_size_max: float = 0.0
    font_weight: Optional[int] = None
    line_spacing: float = 1.2
    letter_spacing: float = 1.15
    text_padding: float = 0.0
    line_break_strategy: str = "auto"
    opacity: float = 1.0
    shadow_radius: float = 0.0
    shadow_strength: float = 1.0
    shadow_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    shadow_offset: List[float] = field(default_factory=lambda: [0.0, 0.0])
    gradient_enabled: bool = False
    gradient_type: int = 0
    gradient_start_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    gradient_end_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    gradient_angle: float = 0.0
    gradient_size: float = 1.0
    text_on_path: int = 0
    text_on_path_arc_degrees: float = 180.0
    warp_style: int = 0
    warp_strength: float = 0.5
    perspective_x: float = 0.0
    perspective_y: float = 0.0
    extrusion_depth: float = 0.0
    blend_mode: int = 0
    auto_fit_font_size: bool = False
    outline_only: bool = False
    stroke_outline_outside_only: bool = False
    overlay_opacity_fmt: float = 1.0
    skew_x_fmt: float = 0.0
    skew_y_fmt: float = 0.0
    text_box_corner_radius: float = 0.0
    text_box_shape: str = ""
    fallback_font_chain: str = ""
    manga_preset: str = ""

    @staticmethod
    def from_textblock(blk: TextBlock) -> "TextBlockState":
        """Snapshot a TextBlock into a flat state object."""
        ff = blk.fontformat
        # Serialize numpy mask to bytes if present
        mask_bytes = None
        if blk.text_mask is not None:
            import numpy as np
            mask_bytes = blk.text_mask.tobytes()

        return TextBlockState(
            xyxy=list(blk.xyxy) if blk.xyxy else [0.0, 0.0, 0.0, 0.0],
            lines=list(blk.lines) if blk.lines else [],
            text=list(blk.text) if blk.text else [],
            translation=blk.translation or "",
            rich_text=blk.rich_text or "",
            angle=blk.angle,
            src_is_vertical=blk.src_is_vertical,
            text_mask=mask_bytes,
            foreground_image_path=blk.foreground_image_path,
            overlay_opacity=blk.overlay_opacity,
            skew_x=blk.skew_x,
            skew_y=blk.skew_y,
            warp_mode=blk.warp_mode,
            warp_quad=[list(p) for p in blk.warp_quad] if blk.warp_quad else [[0, 0], [1, 0], [1, 1], [0, 1]],
            warp_mesh_size=list(blk.warp_mesh_size) if blk.warp_mesh_size else [2, 2],
            # FontFormat
            font_family=ff.font_family or "",
            font_size=ff.font_size,
            stroke_width=ff.stroke_width,
            secondary_stroke_width=ff.secondary_stroke_width,
            frgb=list(ff.frgb) if ff.frgb else [0, 0, 0],
            srgb=list(ff.srgb) if ff.srgb else [0, 0, 0],
            secondary_srgb=list(ff.secondary_srgb) if ff.secondary_srgb else [255, 255, 255],
            bold=ff.bold,
            underline=ff.underline,
            strikethrough=ff.strikethrough,
            italic=ff.italic,
            alignment=ff.alignment,
            vertical=ff.vertical,
            writing_mode=ff.writing_mode,
            fit_mode=ff.fit_mode,
            fit_font_size_min=ff.fit_font_size_min,
            fit_font_size_max=ff.fit_font_size_max,
            font_weight=ff.font_weight,
            line_spacing=ff.line_spacing,
            letter_spacing=ff.letter_spacing,
            text_padding=ff.text_padding,
            line_break_strategy=ff.line_break_strategy,
            opacity=ff.opacity,
            shadow_radius=ff.shadow_radius,
            shadow_strength=ff.shadow_strength,
            shadow_color=list(ff.shadow_color) if ff.shadow_color else [0, 0, 0],
            shadow_offset=list(ff.shadow_offset) if ff.shadow_offset else [0.0, 0.0],
            gradient_enabled=ff.gradient_enabled,
            gradient_type=ff.gradient_type,
            gradient_start_color=list(ff.gradient_start_color) if ff.gradient_start_color else [0, 0, 0],
            gradient_end_color=list(ff.gradient_end_color) if ff.gradient_end_color else [255, 255, 255],
            gradient_angle=ff.gradient_angle,
            gradient_size=ff.gradient_size,
            text_on_path=ff.text_on_path,
            text_on_path_arc_degrees=ff.text_on_path_arc_degrees,
            warp_style=ff.warp_style,
            warp_strength=ff.warp_strength,
            perspective_x=ff.perspective_x,
            perspective_y=ff.perspective_y,
            extrusion_depth=ff.extrusion_depth,
            blend_mode=ff.blend_mode,
            auto_fit_font_size=ff.auto_fit_font_size,
            outline_only=ff.outline_only,
            stroke_outline_outside_only=ff.stroke_outline_outside_only,
            overlay_opacity_fmt=ff.overlay_opacity,
            skew_x_fmt=ff.skew_x,
            skew_y_fmt=ff.skew_y,
            text_box_corner_radius=ff.text_box_corner_radius,
            text_box_shape=ff.text_box_shape,
            fallback_font_chain=ff.fallback_font_chain,
            manga_preset=ff.manga_preset,
        )

    def apply_to_textblock(self, blk: TextBlock) -> None:
        """Restore this snapshot onto a TextBlock."""
        blk.xyxy = list(self.xyxy)
        blk.lines = list(self.lines)
        blk.text = list(self.text)
        blk.translation = self.translation
        blk.rich_text = self.rich_text
        blk.angle = self.angle
        blk.src_is_vertical = self.src_is_vertical
        if self.text_mask is not None:
            import numpy as np
            # Reconstruct mask with same shape as block bbox
            w = int(blk.xyxy[2] - blk.xyxy[0])
            h = int(blk.xyxy[3] - blk.xyxy[1])
            if w > 0 and h > 0 and len(self.text_mask) == w * h:
                blk.text_mask = np.frombuffer(self.text_mask, dtype=np.uint8).reshape((h, w))
            else:
                blk.text_mask = None
        else:
            blk.text_mask = None
        blk.foreground_image_path = self.foreground_image_path
        blk.overlay_opacity = self.overlay_opacity
        blk.skew_x = self.skew_x
        blk.skew_y = self.skew_y
        blk.warp_mode = self.warp_mode
        blk.warp_quad = [list(p) for p in self.warp_quad]
        blk.warp_mesh_size = list(self.warp_mesh_size)

        ff = blk.fontformat
        ff.font_family = self.font_family
        ff.font_size = self.font_size
        ff.stroke_width = self.stroke_width
        ff.secondary_stroke_width = self.secondary_stroke_width
        ff.frgb = list(self.frgb)
        ff.srgb = list(self.srgb)
        ff.secondary_srgb = list(self.secondary_srgb)
        ff.bold = self.bold
        ff.underline = self.underline
        ff.strikethrough = self.strikethrough
        ff.italic = self.italic
        ff.alignment = self.alignment
        ff.vertical = self.vertical
        ff.writing_mode = self.writing_mode
        ff.fit_mode = self.fit_mode
        ff.fit_font_size_min = self.fit_font_size_min
        ff.fit_font_size_max = self.fit_font_size_max
        ff.font_weight = self.font_weight
        ff.line_spacing = self.line_spacing
        ff.letter_spacing = self.letter_spacing
        ff.text_padding = self.text_padding
        ff.line_break_strategy = self.line_break_strategy
        ff.opacity = self.opacity
        ff.shadow_radius = self.shadow_radius
        ff.shadow_strength = self.shadow_strength
        ff.shadow_color = list(self.shadow_color)
        ff.shadow_offset = list(self.shadow_offset)
        ff.gradient_enabled = self.gradient_enabled
        ff.gradient_type = self.gradient_type
        ff.gradient_start_color = list(self.gradient_start_color)
        ff.gradient_end_color = list(self.gradient_end_color)
        ff.gradient_angle = self.gradient_angle
        ff.gradient_size = self.gradient_size
        ff.text_on_path = self.text_on_path
        ff.text_on_path_arc_degrees = self.text_on_path_arc_degrees
        ff.warp_style = self.warp_style
        ff.warp_strength = self.warp_strength
        ff.perspective_x = self.perspective_x
        ff.perspective_y = self.perspective_y
        ff.extrusion_depth = self.extrusion_depth
        ff.blend_mode = self.blend_mode
        ff.auto_fit_font_size = self.auto_fit_font_size
        ff.outline_only = self.outline_only
        ff.stroke_outline_outside_only = self.stroke_outline_outside_only
        ff.overlay_opacity = self.overlay_opacity_fmt
        ff.skew_x = self.skew_x_fmt
        ff.skew_y = self.skew_y_fmt
        ff.text_box_corner_radius = self.text_box_corner_radius
        ff.text_box_shape = self.text_box_shape
        ff.fallback_font_chain = self.fallback_font_chain
        ff.manga_preset = self.manga_preset


class TextBlockStateCommand(QUndoCommand):
    """
    State-based undo command for a single TextBlock.

    Usage:
        old_state = TextBlockState.from_textblock(blk)
        # ... mutate blk ...
        new_state = TextBlockState.from_textblock(blk)
        cmd = TextBlockStateCommand(blk, old_state, new_state, "Change font size")
        undo_stack.push(cmd)
    """

    def __init__(
        self,
        blk: TextBlock,
        old_state: TextBlockState,
        new_state: TextBlockState,
        description: str = "Edit text block",
        item=None,  # Optional TextBlkItem to update visuals
    ):
        super().__init__(description)
        self.blk = blk
        self.old_state = old_state
        self.new_state = new_state
        self.item = item

    def undo(self):
        self.old_state.apply_to_textblock(self.blk)
        if self.item is not None:
            self.item.update_from_blk()
            self.item.update()

    def redo(self):
        self.new_state.apply_to_textblock(self.blk)
        if self.item is not None:
            self.item.update_from_blk()
            self.item.update()


class MultiTextBlockStateCommand(QUndoCommand):
    """
    State-based undo command for multiple text blocks (batch edit).
    """

    def __init__(
        self,
        snapshots: List[Tuple[TextBlock, TextBlockState, TextBlockState]],
        description: str = "Batch edit",
        items=None,
    ):
        super().__init__(description)
        # List of (blk, old_state, new_state)
        self.snapshots = snapshots
        self.items = items or {}

    def undo(self):
        for blk, old, _ in self.snapshots:
            old.apply_to_textblock(blk)
            item = self.items.get(id(blk))
            if item is not None:
                item.update_from_blk()
                item.update()

    def redo(self):
        for blk, _, new in self.snapshots:
            new.apply_to_textblock(blk)
            item = self.items.get(id(blk))
            if item is not None:
                item.update_from_blk()
                item.update()

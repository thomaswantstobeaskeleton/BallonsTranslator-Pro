from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from qtpy.QtCore import QObject


@dataclass
class ActionRecord:
    action_id: str
    top_level: str
    label: str
    menu_path: str
    category: str
    workflow_mode: str
    simple_mode_visible: bool
    advanced_mode_visible: bool
    command_palette_visible: bool
    shortcut: str
    tooltip: str
    icon_name: str = ""
    context_menu_category: str = ""
    danger_level: str = "normal"
    enabled: bool = True
    disabled_reason: str = ""
    action: object = None


class ActionRegistry(QObject):
    """Central registry for discoverable UI actions.

    First milestone scope: register existing menu actions without behavior changes.
    """

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._records: Dict[str, ActionRecord] = {}
        self._action_to_id: Dict[int, str] = {}

    @staticmethod
    def _slug(text: str) -> str:
        out = []
        for ch in (text or "").lower():
            if ch.isalnum():
                out.append(ch)
            elif ch in (" ", "-", "/", ":", ".", "(", ")"):
                out.append("_")
        return "_".join(part for part in "".join(out).split("_") if part)

    def _build_id(self, top_level: str, path: str, label: str) -> str:
        return f"{self._slug(top_level)}.{self._slug(path)}.{self._slug(label)}"

    def register_qaction(
        self,
        *,
        top_level: str,
        menu_path: str,
        qaction,
        category: Optional[str] = None,
        workflow_mode: str = "editor",
        simple_mode_visible: bool = True,
        advanced_mode_visible: bool = True,
        command_palette_visible: bool = True,
    ) -> ActionRecord:
        text = (qaction.text() or "").replace("&", "").strip()
        aid = self._build_id(top_level, menu_path, text)
        shortcut = qaction.shortcut().toString() if hasattr(qaction, "shortcut") else ""
        tooltip = qaction.toolTip() if hasattr(qaction, "toolTip") else ""
        icon_name = ""
        if hasattr(qaction, "icon") and not qaction.icon().isNull():
            icon_name = "icon"
        enabled = True
        if hasattr(qaction, "isEnabled"):
            try:
                enabled = bool(qaction.isEnabled())
            except Exception:
                enabled = True

        rec = ActionRecord(
            action_id=aid,
            top_level=str(top_level or ""),
            label=text,
            menu_path=menu_path,
            category=category or top_level,
            workflow_mode=workflow_mode,
            simple_mode_visible=simple_mode_visible,
            advanced_mode_visible=advanced_mode_visible,
            command_palette_visible=command_palette_visible,
            shortcut=shortcut,
            tooltip=tooltip,
            icon_name=icon_name,
            enabled=enabled,
            disabled_reason="Action is currently unavailable." if not enabled else "",
            action=qaction,
        )
        self._records[aid] = rec
        self._action_to_id[id(qaction)] = aid
        return rec

    def register_menu_tree(self, top_level: str, menu, *, menu_path_prefix: str = "") -> None:
        if menu is None:
            return
        for act in menu.actions():
            if act.isSeparator():
                continue
            sub = act.menu()
            if sub is not None:
                title = (sub.title() or act.text() or "").replace("&", "").strip()
                next_path = f"{menu_path_prefix}{title}"
                self.register_menu_tree(top_level, sub, menu_path_prefix=f"{next_path} > ")
                continue
            self.register_qaction(
                top_level=top_level,
                menu_path=(menu_path_prefix[:-3] if menu_path_prefix.endswith(" > ") else menu_path_prefix) or top_level,
                qaction=act,
            )

    def clear(self) -> None:
        self._records.clear()
        self._action_to_id.clear()

    def record_for_action(self, action) -> Optional[ActionRecord]:
        aid = self._action_to_id.get(id(action))
        if not aid:
            return None
        return self._records.get(aid)

    def all_records(self) -> List[ActionRecord]:
        return list(self._records.values())

    def discoverable_actions(self, show_unavailable: bool = False) -> Iterable[Tuple[str, object, dict]]:
        for rec in self._records.values():
            if not rec.command_palette_visible or rec.action is None:
                continue
            if (not show_unavailable) and (not rec.enabled):
                continue
            label = f"[Command] Menu > {rec.menu_path} > {rec.label}" if rec.menu_path else f"[Command] Menu > {rec.label}"
            extra = ""
            if rec.tooltip:
                extra += f" ({rec.tooltip})"
            if rec.shortcut:
                extra += f" [{rec.shortcut}]"
            if not rec.enabled:
                extra += f" [Unavailable: {rec.disabled_reason}]"
            yield (label + extra, rec.action, {"enabled": rec.enabled, "reason": rec.disabled_reason})

    def has_duplicate_ids(self) -> bool:
        return len(self._records) != len(set(self._records.keys()))

    def to_rows(self) -> List[dict]:
        rows = []
        for rec in self._records.values():
            rows.append({
                "action_id": rec.action_id,
                "top_level": rec.top_level,
                "label": rec.label,
                "menu_path": rec.menu_path,
                "category": rec.category,
                "workflow_mode": rec.workflow_mode,
                "simple_mode_visible": rec.simple_mode_visible,
                "advanced_mode_visible": rec.advanced_mode_visible,
                "command_palette_visible": rec.command_palette_visible,
                "shortcut": rec.shortcut,
                "tooltip": rec.tooltip,
                "enabled": rec.enabled,
                "disabled_reason": rec.disabled_reason,
                "danger_level": rec.danger_level,
            })
        return rows

    def summary_stats(self) -> dict:
        rows = self.to_rows()
        total = len(rows)
        enabled = sum(1 for r in rows if r.get("enabled"))
        simple_visible = sum(1 for r in rows if r.get("simple_mode_visible"))
        advanced_visible = sum(1 for r in rows if r.get("advanced_mode_visible"))
        by_category: Dict[str, int] = {}
        for r in rows:
            c = str(r.get("category") or "Uncategorized")
            by_category[c] = by_category.get(c, 0) + 1
        return {
            "total_actions": total,
            "enabled_actions": enabled,
            "disabled_actions": max(0, total - enabled),
            "simple_visible_actions": simple_visible,
            "advanced_visible_actions": advanced_visible,
            "categories": by_category,
            "duplicate_shortcuts": self.duplicate_shortcuts(),
        }

    def duplicate_shortcuts(self) -> Dict[str, List[str]]:
        by_shortcut: Dict[str, List[str]] = {}
        for rec in self._records.values():
            if not rec.shortcut:
                continue
            by_shortcut.setdefault(rec.shortcut, []).append(rec.action_id)
        return {k: v for k, v in by_shortcut.items() if len(v) > 1}

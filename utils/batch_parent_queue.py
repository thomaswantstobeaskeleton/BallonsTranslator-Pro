from __future__ import annotations

import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _dir_has_images(d: str) -> bool:
    try:
        for name in os.listdir(d):
            if osp.splitext(name)[1].lower() in IMG_EXT:
                return True
    except OSError:
        return False
    return False


def _iter_image_dirs(root_dir: str) -> List[str]:
    out: List[str] = []
    root_dir = osp.normpath(osp.abspath(root_dir))
    for cur, dirnames, _f in os.walk(root_dir):
        base = osp.basename(cur).lower()
        if base in {"result", "mask", "inpainted"}:
            dirnames[:] = []
            continue
        if _dir_has_images(cur):
            out.append(osp.normpath(osp.abspath(cur)))
    out.sort(key=lambda p: (p.count(os.sep), p.lower()))
    return out


@dataclass
class BatchChildProject:
    kind: str  # dir|cbz|zip
    input_path: str
    display_name: str


def enumerate_child_projects(parent_path: str) -> List[BatchChildProject]:
    parent_path = osp.abspath(parent_path)
    out: List[BatchChildProject] = []
    if osp.isdir(parent_path):
        # nested image dirs
        for d in _iter_image_dirs(parent_path):
            out.append(BatchChildProject(kind='dir', input_path=d, display_name=osp.relpath(d, parent_path)))
        # archive files in tree
        for cur, _dirs, files in os.walk(parent_path):
            for name in files:
                low = name.lower()
                if low.endswith('.cbz') or low.endswith('.zip'):
                    p = osp.abspath(osp.join(cur, name))
                    out.append(BatchChildProject(kind='cbz' if low.endswith('.cbz') else 'zip', input_path=p, display_name=osp.relpath(p, parent_path)))
    else:
        low = parent_path.lower()
        if low.endswith('.cbz') or low.endswith('.zip'):
            out.append(BatchChildProject(kind='cbz' if low.endswith('.cbz') else 'zip', input_path=parent_path, display_name=osp.basename(parent_path)))
    # stable unique
    dedup: Dict[str, BatchChildProject] = {}
    for c in out:
        dedup[c.input_path] = c
    return [dedup[k] for k in sorted(dedup.keys())]


def save_parent_batch_state(state_path: str, parent_path: str, children: List[BatchChildProject], *, statuses: Dict[str, str] | None = None) -> Dict[str, object]:
    statuses = dict(statuses or {})
    payload = {
        'format': 'ballonstranslator.parent_batch_state.v1',
        'parent_path': osp.abspath(parent_path),
        'children': [
            {
                'kind': c.kind,
                'input_path': c.input_path,
                'display_name': c.display_name,
                'status': statuses.get(c.input_path, 'pending'),
            }
            for c in children
        ],
    }
    os.makedirs(osp.dirname(osp.abspath(state_path)) or '.', exist_ok=True)
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def load_parent_batch_state(state_path: str) -> dict:
    with open(state_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get('format') != 'ballonstranslator.parent_batch_state.v1':
        raise ValueError('invalid_state_format')
    return payload


def update_parent_batch_status(state_path: str, input_path: str, status: str) -> dict:
    payload = load_parent_batch_state(state_path)
    target = osp.abspath(input_path)
    updated = False
    for row in payload.get('children', []) or []:
        if osp.abspath(str(row.get('input_path', '') or '')) == target:
            row['status'] = str(status or '').strip() or 'pending'
            updated = True
            break
    if not updated:
        raise ValueError('child_not_found')
    with open(state_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload


def next_pending_child(state_payload: dict) -> dict | None:
    for row in list((state_payload or {}).get('children') or []):
        if str((row or {}).get('status', 'pending') or 'pending') == 'pending':
            return dict(row)
    return None


def summarize_parent_batch_state(state_payload: dict) -> dict:
    rows = list((state_payload or {}).get('children') or [])
    by_status: Dict[str, int] = {}
    for row in rows:
        st = str((row or {}).get('status', 'pending') or 'pending').strip() or 'pending'
        by_status[st] = by_status.get(st, 0) + 1
    return {
        'format': str((state_payload or {}).get('format', '') or ''),
        'parent_path': str((state_payload or {}).get('parent_path', '') or ''),
        'total': len(rows),
        'counts': by_status,
        'next_pending': next_pending_child(state_payload),
    }

from pathlib import Path

from utils.batch_parent_queue import enumerate_child_projects, save_parent_batch_state, load_parent_batch_state, update_parent_batch_status, next_pending_child


def test_enumerate_child_projects_collects_dirs_and_archives(tmp_path: Path):
    root = tmp_path / 'root'
    (root / 'ch1').mkdir(parents=True)
    (root / 'ch1' / '001.png').write_bytes(b'x')
    (root / 'pack.cbz').write_bytes(b'zip')
    children = enumerate_child_projects(str(root))
    kinds = sorted([c.kind for c in children])
    assert 'dir' in kinds
    assert 'cbz' in kinds


def test_save_parent_batch_state_writes_statuses(tmp_path: Path):
    root = tmp_path / 'root'
    root.mkdir()
    child = enumerate_child_projects(str(root))
    if not child:
        from utils.batch_parent_queue import BatchChildProject
        child = [BatchChildProject(kind='dir', input_path=str(root), display_name='.')]
    state = save_parent_batch_state(str(tmp_path / 'state.json'), str(root), child, statuses={child[0].input_path: 'done'})
    assert state['children'][0]['status'] == 'done'


def test_parent_state_load_update_and_next_pending(tmp_path: Path):
    root = tmp_path / 'root'
    (root / 'ch1').mkdir(parents=True)
    (root / 'ch1' / '001.png').write_bytes(b'x')
    children = enumerate_child_projects(str(root))
    state_path = tmp_path / 'state.json'
    save_parent_batch_state(str(state_path), str(root), children)
    loaded = load_parent_batch_state(str(state_path))
    assert loaded['format'].endswith('v1')
    first = loaded['children'][0]['input_path']
    update_parent_batch_status(str(state_path), first, 'done')
    loaded2 = load_parent_batch_state(str(state_path))
    nxt = next_pending_child(loaded2)
    if len(loaded2['children']) == 1:
        assert nxt is None
    else:
        assert nxt is not None

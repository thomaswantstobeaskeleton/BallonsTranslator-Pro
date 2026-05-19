from pathlib import Path

from utils.archive_stream_export import write_archive_streaming


def test_archive_streaming_writes_progress_and_archive(tmp_path: Path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.txt').write_text('a', encoding='utf-8')
    (src / 'b.txt').write_text('b', encoding='utf-8')
    out = tmp_path / 'out.zip'
    rst = write_archive_streaming(str(src), str(out))
    assert out.exists()
    assert rst['written_files'] == 2
    assert rst['progress_events'][-1]['progress'] == 1.0


def test_archive_streaming_can_cancel(tmp_path: Path):
    src = tmp_path / 'src2'
    src.mkdir()
    for i in range(4):
        (src / f'{i}.txt').write_text(str(i), encoding='utf-8')
    out = tmp_path / 'out2.zip'
    called = {'n': 0}

    def cancel_check():
        called['n'] += 1
        return called['n'] > 2

    rst = write_archive_streaming(str(src), str(out), cancel_check=cancel_check)
    assert rst['cancelled'] is True
    assert rst['written_files'] < rst['total_files']


def test_archive_streaming_progress_hook_receives_events(tmp_path: Path):
    src = tmp_path / 'src3'
    src.mkdir()
    (src / 'a.txt').write_text('a', encoding='utf-8')
    (src / 'b.txt').write_text('b', encoding='utf-8')
    out = tmp_path / 'out3.zip'
    events = []
    rst = write_archive_streaming(str(src), str(out), progress_hook=lambda ev: events.append(ev))
    assert rst['written_files'] == 2
    assert len(events) == 2
    assert events[-1]['progress'] == 1.0

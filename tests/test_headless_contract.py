from utils.headless_contract import HeadlessRunSummary, HeadlessExitCode


def test_headless_summary_ok_exit_code():
    s = HeadlessRunSummary(requested_dirs=['a'], processed_dirs=['a'], skipped_dirs=[], failed_dirs=[])
    assert s.exit_code() == int(HeadlessExitCode.OK)


def test_headless_summary_partial_failure_exit_code():
    s = HeadlessRunSummary(requested_dirs=['a', 'b'], processed_dirs=['a'], skipped_dirs=[], failed_dirs=['b'])
    assert s.exit_code() == int(HeadlessExitCode.PARTIAL_FAILURE)


def test_headless_summary_payload_counts():
    s = HeadlessRunSummary(requested_dirs=['a', 'b'], processed_dirs=[], skipped_dirs=['a'], failed_dirs=['b'])
    payload = s.to_payload()
    assert payload['requested'] == 2
    assert payload['processed'] == 0
    assert payload['skipped'] == 1
    assert payload['failed'] == 1


def test_headless_summary_payload_includes_warnings():
    s = HeadlessRunSummary(requested_dirs=['a'], processed_dirs=['a'], skipped_dirs=[], failed_dirs=[], warnings=['w1'])
    payload = s.to_payload()
    assert payload['warnings'] == ['w1']

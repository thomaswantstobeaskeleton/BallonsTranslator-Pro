from utils.automation_jobs import new_job, checkpoint_or_cancel, set_status, status_payload, append_log, update_from_task_result


def test_job_lifecycle_progress_and_status_payload():
    job = new_job('job_1', 'run_pipeline')
    set_status(job, 'running', stage='ocr', progress=0.35)
    payload = status_payload(job)
    assert payload['status'] == 'running'
    assert payload['stage'] == 'ocr'
    assert abs(payload['progress'] - 0.35) < 1e-9


def test_checkpoint_cancels_when_requested():
    job = new_job('job_2', 'export')
    job['cancel_requested'] = True
    cancelled = checkpoint_or_cancel(job, 'render', 0.6)
    assert cancelled is True
    assert job['status'] == 'cancelled'
    assert 'cancelled at checkpoint' in '\n'.join(job['logs'])


def test_append_log_respects_limit():
    job = new_job('job_3', 'proof_pack')
    for i in range(10):
        append_log(job, f'line-{i}', limit=3)
    assert len(job['logs']) == 3
    assert job['logs'][0] == 'line-7'


def test_update_from_task_result_captures_warnings_and_progress():
    job = new_job('job_4', 'export')
    update_from_task_result(job, {'warnings': ['missing font'], 'stage': 'exporting', 'progress': 0.7})
    assert job['warnings'] == ['missing font']
    assert job['stage'] == 'exporting'
    assert abs(job['progress'] - 0.7) < 1e-9

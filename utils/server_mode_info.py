from __future__ import annotations

import os
import os.path as osp
from typing import Dict, Any


def build_server_mode_info(*, host: str = '127.0.0.1', port: int = 39542) -> Dict[str, Any]:
    program_path = osp.abspath(os.getcwd())
    data_path = osp.abspath('data')
    config_path = osp.abspath(osp.join('config', 'config.json'))
    return {
        'host': host,
        'port': int(port),
        'health_url': f'http://{host}:{int(port)}/health',
        'routes_url': f'http://{host}:{int(port)}/routes',
        'events_template': f'http://{host}:{int(port)}/events?job_id=<job_id>',
        'paths': {
            'program_path': program_path,
            'data_path': data_path,
            'config_path': config_path,
        },
        'docker_mount_hints': {
            'models_cache': '/app/data/models',
            'projects': '/app/projects',
            'config': '/app/config/config.json',
        },
        'sample_curl': [
            f'curl http://{host}:{int(port)}/health',
            f"curl -X POST http://{host}:{int(port)}/project_status -H 'Content-Type: application/json' -d '{{}}'",
        ],
        'sample_python': (
            "import requests\n"
            f"base='http://{host}:{int(port)}'\n"
            "print(requests.get(base+'/health', timeout=5).json())\n"
            "print(requests.get(base+'/routes', timeout=5).json())\n"
        ),
    }

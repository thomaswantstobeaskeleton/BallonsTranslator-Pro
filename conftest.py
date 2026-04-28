"""Pytest configuration for this repository.

Some files named `test_*.py` are utility scripts/helpers rather than pytest
test modules and require runtime parameters (not pytest fixtures). Exclude them
from pytest collection so CI/local `pytest` runs execute only real tests.
"""

collect_ignore = [
    "scripts/test_manga_sources.py",
    "utils/translator_test.py",
]

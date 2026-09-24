import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True, scope="session")
def _isolate_results_dir(tmp_path_factory):
    """Redirect save_result() to a temporary directory for the whole test run.

    results/ is the data of record; a test that calls a phase function (e.g.
    phase7_scaling(quick=True)) must never overwrite it or add history files.
    """
    from qmembench import experiments
    original = experiments.RESULTS_DIR
    experiments.RESULTS_DIR = tmp_path_factory.mktemp("results")
    yield
    experiments.RESULTS_DIR = original

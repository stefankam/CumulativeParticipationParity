import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "server"))
from baselines import (BASELINE_SOURCES, BaselineUnavailableError, fedavg, qfedavg,
                       require_reference_baseline)


def test_fedavg_uses_sample_counts():
    result = fedavg({"w": 0.0}, [{"w": 2.0}, {"w": 6.0}], [1, 3])
    assert result == {"w": 5.0}


def test_qfedavg_matches_closed_form_server_update():
    # delta=(1, 2), L=(1,2), q=1, eta=1 => weighted deltas=(1,4), h=(2,6)
    result = qfedavg({"w": 4.0}, [{"w": 3.0}, {"w": 2.0}], [1.0, 2.0], 1.0, 1.0)
    assert math.isclose(result["w"], 4.0 - 5.0 / 8.0)


def test_unverified_named_baselines_fail_instead_of_using_placeholders():
    with pytest.raises(BaselineUnavailableError):
        require_reference_baseline("php_fl")
    with pytest.raises(BaselineUnavailableError):
        require_reference_baseline("fairfedcs")
    assert BASELINE_SOURCES["php_fl"].code_url.endswith("Siyuan01/PHP-FL-main")
    assert "openreview.net/forum?id=pJWozQn9p4" in BASELINE_SOURCES["php_fl"].paper_url
    assert BASELINE_SOURCES["fairfedcs"].paper_url.endswith("2307.10738")

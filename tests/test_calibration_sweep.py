"""Tests for the standalone Calvano calibration parameter sweep."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


CALIBRATION_PATH = Path(__file__).parents[1] / "calvano-sourcecode" / "calibration.py"


def _load_calibration_module():
    spec = importlib.util.spec_from_file_location("calibration_sweep", CALIBRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parameter_sweep_derives_mu_for_each_outside_share() -> None:
    calibration = _load_calibration_module()
    parameters = {
        "A": 1.0,
        "ABS_EPSILON": 0.6,
        "C": 0.75,
        "P_OUTSIDE": [0.10, 0.30, 0.50],
        "S_MIN": 0.5,
        "S_MAX": 5.0,
    }

    cases = calibration.parameter_sets(parameters, "P_OUTSIDE")
    results = [calibration.run_calibration(case) for case in cases]

    assert [case["P_OUTSIDE"] for case in cases] == [0.10, 0.30, 0.50]
    assert [result["parameters"]["MU"] for result in results] == pytest.approx(
        [(1 + p_outside) / (2 * 0.6) for p_outside in [0.10, 0.30, 0.50]]
    )
    assert len(results) == 3
    assert all(
        result["baseline_shares"][2] == pytest.approx(result["parameters"]["P_OUTSIDE"])
        for result in results
    )


def test_parameter_sweep_requires_explicit_list_marker() -> None:
    calibration = _load_calibration_module()
    parameters = {
        "A": 1.0,
        "ABS_EPSILON": 0.6,
        "C": 0.75,
        "P_OUTSIDE": [0.20, 0.25],
        "S_MIN": 0.5,
        "S_MAX": 5.0,
    }

    with pytest.raises(ValueError, match="set SWEEP_PARAMETER"):
        calibration.parameter_sets(parameters)

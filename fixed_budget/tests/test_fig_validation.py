# Author: Manan Gupta <mnn@yogins.com>
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import fig_validation as fv


def test_time_helpers_on_an_exponential():
    t = np.geomspace(0.01, 50, 400)
    c = np.exp(-t / 3.0)
    assert abs(fv.efold(t, c) - 3.0) < 0.01
    assert abs(fv.fit_memory(t, c) - 3.0) < 1e-6
    assert abs(fv.at(t, c, 3.0) - np.exp(-1)) < 1e-3


def test_cosh_ratio():
    assert fv.cosh_ratio([2.0, 0.0], [4.0, 0.0], [0.0, 0.0]) == [0.5, 0.0]
    r, e = fv.cosh_ratio([1.0, 0.1], [2.0, 0.0], [1.0, 0.0])
    assert np.isclose(r, np.cosh(1.0) / 2) and np.isclose(e, r * 0.1)


@pytest.mark.skipif(not (fv.MUMAX / "device_p_equilibrium.json").exists()
                    or not (fv.SLLG / "device.json").exists(), reason="stored records missing")
def test_build_from_stored_records(tmp_path, monkeypatch):
    monkeypatch.setattr(fv, "OUT", tmp_path)
    fv.build(fv.MUMAX, fv.SLLG, "check")
    v = json.loads((tmp_path / "check.json").read_text())
    mm, ms = v["mumax"], v["macrospin"]
    # Each device is on its own time scale, so zero bias sits at 1.
    zero = next(p for p in mm["tau_vs_I"] if p["b_mT"] == 0)
    assert zero["tau_over_tau0"] == [1.0, 0.0]
    zero = next(p for p in ms["tau_vs_I"] if p["b_mT"] == 0)
    assert np.isclose(zero["tau_over_tau0"][0], 1.0, rtol=0.05)
    reads = v["compare"]["reads"]["0"]
    assert [r["r0"] for r in reads] == fv.R0_GRID
    # Only the zero-bias row at r0 = 1 is fixed by the definition of tau_c(0).
    assert [r["r0"] for r in reads if r["by_construction"]] == [1]
    assert not any(r["by_construction"] for r in v["compare"]["reads"]["18"])
    for r in reads:
        assert np.isclose(r["telegraph_acf"][0], np.exp(-r["r0"]), rtol=0.01)
        assert -0.05 < r["mumax_acf"][0] < 1 and -0.05 < r["macrospin_acf"] < 1
    assert v["compare"]["tau_vs_I"] and v["compare"]["steps"]
    for s in mm["steps"]:
        # A measured I is one number; an extrapolated one is a range with its lower bound.
        assert (len(s["I_options"]) == 1) == (s["I_source"] != "field_fit")
        half = fv.extrapolation_span(s)[1]
        assert half >= 0 and (half > 0) == (s["I_source"] == "field_fit")
        for label, i in s["I_options"].items():
            assert np.isclose(s["tau_over_cosh"][label][0], s["tau_over_tau0"][0] * np.cosh(i))


def test_ratio_carries_both_errors():
    r, e = fv.ratio([2.0, 0.2], [4.0, 0.4])
    assert np.isclose(r, 0.5) and np.isclose(e, 0.5 * np.hypot(0.1, 0.1))
    assert fv.ratio([2.0, None], [4.0, 0.0]) == [0.5, 0.0]

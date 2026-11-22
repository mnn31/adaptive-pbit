# Author: Manan Gupta <mnn@yogins.com>
"""The letter's figure script draws Figs. 2 and 3 from the stored records."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "fig_letter.py"
TABLE = ROOT / "results" / "sllg_matched" / "classifier_table.json"


@pytest.mark.skipif(not TABLE.exists() or not (ROOT / "results" / "v2").exists(),
                    reason="stored records missing")
def test_figures_are_drawn(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), str(tmp_path), "fig2", "fig3", "dispersion"],
                   check=True)
    for name in ["gupta2", "gupta3", "dispersion"]:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 5000

"""Writers compatible with the supplied Figure 4 R script."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_figure4_input(path: str | Path, summary: dict[str, np.ndarray | float]) -> Path:
    """Write the whitespace-delimited ``A_irToBR.txt`` consumed by Figure 4."""
    target = Path(path)
    dev = np.asarray(summary["AggrDevPriceShock"], dtype=float)
    non_dev = np.asarray(summary["AggrNonDevPriceShock"], dtype=float)
    fields = ["AggrPricePre"]
    fields.extend(f"AggrDevPriceShockPer{period:03d}" for period in range(1, len(dev) + 1))
    fields.extend(f"AggrNonDevPriceShockPer{period:03d}" for period in range(1, len(non_dev) + 1))
    values = [float(summary["AggrPricePre"]), *dev.tolist(), *non_dev.tolist()]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(" ".join(fields) + "\n" + " ".join(f"{value:.7f}" for value in values) + "\n", encoding="utf-8")
    return target

from __future__ import annotations

import logging
from typing import Optional

from pipeline.estimator import ROISizeEstimate

logger = logging.getLogger(__name__)


def select_rois_cli(
    estimates: list[ROISizeEstimate],
    roi_ids: Optional[list[str]] = None,
    proportion: Optional[float] = None,
    max_rois: Optional[int] = None,
    max_total_size_gb: Optional[float] = None,
    select_all: bool = False,
) -> list[ROISizeEstimate]:
    """Select ROIs based on CLI flags. Returns the filtered list."""
    viable = [e for e in estimates if e.n_tiles_ok > 0]

    if not viable:
        logger.warning("[selector] No viable ROIs (all tiles failed)")
        return []

    if roi_ids is not None:
        id_set = set(roi_ids)
        viable = [e for e in viable if e.roi_acquisition_id in id_set]
        missing = id_set - {e.roi_acquisition_id for e in viable}
        if missing:
            logger.warning("[selector] Requested ROI IDs not found: %s", sorted(missing))

    if select_all:
        selected = viable
    elif proportion is not None:
        n = max(1, int(len(viable) * proportion))
        selected = viable[:n]
    elif max_rois is not None:
        selected = viable[:max_rois]
    elif max_total_size_gb is not None:
        selected = []
        total = 0.0
        for e in viable:
            if total + e.estimated_processed_size_gb > max_total_size_gb:
                break
            selected.append(e)
            total += e.estimated_processed_size_gb
    else:
        selected = select_rois_interactive(viable)

    logger.info("[selector] selected %d / %d viable ROIs", len(selected), len(viable))
    return selected


def select_rois_interactive(estimates: list[ROISizeEstimate]) -> list[ROISizeEstimate]:
    """Interactive selection prompt."""
    print(f"\n{len(estimates)} viable ROIs available.")
    print("[a]ll  |  [s]elect by index  |  [p]roportion  |  [q]uit")

    while True:
        choice = input("\nSelection> ").strip().lower()

        if choice == "a":
            return estimates

        if choice == "q":
            return []

        if choice == "p":
            try:
                p = float(input("Proportion (0-1)> ").strip())
                n = max(1, int(len(estimates) * p))
                print(f"Selecting first {n} ROIs")
                return estimates[:n]
            except ValueError:
                print("Invalid number. Try again.")
                continue

        if choice == "s":
            raw = input("Indices (e.g. 1,3,5-8)> ").strip()
            indices = _parse_index_spec(raw, len(estimates))
            if indices is None:
                print("Invalid index spec. Try again.")
                continue
            return [estimates[i] for i in sorted(indices)]

        print("Unknown choice. Try again.")


def _parse_index_spec(spec: str, n: int) -> Optional[set[int]]:
    """Parse '1,3,5-8' into a set of 0-based indices."""
    indices: set[int] = set()
    try:
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                lo_i, hi_i = int(lo) - 1, int(hi) - 1
                if lo_i < 0 or hi_i >= n or lo_i > hi_i:
                    return None
                indices.update(range(lo_i, hi_i + 1))
            else:
                i = int(part) - 1
                if i < 0 or i >= n:
                    return None
                indices.add(i)
        return indices if indices else None
    except ValueError:
        return None

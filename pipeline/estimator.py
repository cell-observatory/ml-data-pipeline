from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from typing import Optional

from pipeline.scanner import ROIScanResult

logger = logging.getLogger(__name__)

DESKEW_FACTOR = 0.872
BYTES_PER_VOXEL = 2  # uint16
GB = 1 << 30


@dataclass
class ROISizeEstimate:
    roi_acquisition_id: str
    roi_path: str
    n_tiles_ok: int
    n_tiles_failed: int
    n_channels: int
    max_timepoints: int
    raw_shape: Optional[tuple[int, int, int]]  # z, y, x
    raw_size_gb: float
    estimated_processed_size_gb: float
    channel_time_sizes: dict[str, int]
    failed_tiles: list[tuple[str, list[str]]]


def estimate_roi_size(
    scan_result: ROIScanResult,
    n_channels: int,
) -> ROISizeEstimate:
    """Estimate raw and processed data sizes for an ROI.

    Raw size is computed from image dimensions.
    Processed (deskewed) size is estimated as raw_size / DESKEW_FACTOR.
    """
    ok_tiles = {k: v for k, v in scan_result.tiles.items() if v.status == "ok"}
    failed_tiles = [
        (k, v.error_messages) for k, v in scan_result.tiles.items() if v.status == "failed"
    ]

    raw_shape = scan_result.image_shape

    max_tp = 0
    channel_time_sizes: dict[str, int] = {}
    for tile in ok_tiles.values():
        max_tp = max(max_tp, tile.n_timepoints)
        for ch, ts in tile.time_size.items():
            channel_time_sizes[ch] = max(channel_time_sizes.get(ch, 0), ts)

    n_ok = len(ok_tiles)
    raw_size_gb = 0.0

    if raw_shape and n_ok > 0:
        raw_voxels = raw_shape[0] * raw_shape[1] * raw_shape[2]
        raw_size_gb = n_ok * max_tp * n_channels * raw_voxels * BYTES_PER_VOXEL / GB

    estimated_processed_size_gb = raw_size_gb / DESKEW_FACTOR if raw_size_gb > 0 else 0.0

    return ROISizeEstimate(
        roi_acquisition_id=scan_result.roi_acquisition_id,
        roi_path=scan_result.roi_path,
        n_tiles_ok=n_ok,
        n_tiles_failed=len(failed_tiles),
        n_channels=n_channels,
        max_timepoints=max_tp,
        raw_shape=raw_shape,
        raw_size_gb=round(raw_size_gb, 3),
        estimated_processed_size_gb=round(estimated_processed_size_gb, 3),
        channel_time_sizes=channel_time_sizes,
        failed_tiles=failed_tiles,
    )


def check_disk_capacity(output_folder: str, required_gb: float) -> tuple[bool, float]:
    """Check if output_folder has enough free space.

    Returns (has_space, free_gb).
    """
    try:
        usage = shutil.disk_usage(output_folder)
        free_gb = usage.free / GB
        return (free_gb >= required_gb, round(free_gb, 2))
    except OSError as e:
        logger.warning("Could not check disk usage for %s: %s", output_folder, e)
        return (True, -1.0)


def format_dry_run_summary(
    estimates: list[ROISizeEstimate],
    output_folder: str = "",
) -> str:
    """Produce a formatted summary table for dry-run display."""
    lines = []
    lines.append("=" * 100)
    lines.append("DRY-RUN SUMMARY")
    lines.append("=" * 100)
    lines.append(
        f"{'ROI Path':<50} {'Ch':>3} {'TPs':>5} {'Tiles':>6} {'Failed':>6} "
        f"{'RawShape':>18} {'RawGB':>8} {'ProcGB':>8}"
    )
    lines.append("-" * 100)

    total_raw = 0.0
    total_proc = 0.0
    total_tiles_ok = 0
    total_tiles_failed = 0

    for est in estimates:
        roi_short = est.roi_path
        if len(roi_short) > 48:
            roi_short = "..." + roi_short[-45:]

        tp_note = str(est.max_timepoints)
        if est.channel_time_sizes and len(set(est.channel_time_sizes.values())) > 1:
            tp_note += "*"

        raw_shape_str = (
            f"{est.raw_shape[0]}x{est.raw_shape[1]}x{est.raw_shape[2]}"
            if est.raw_shape else "N/A"
        )

        lines.append(
            f"{roi_short:<50} {est.n_channels:>3} {tp_note:>5} {est.n_tiles_ok:>6} "
            f"{est.n_tiles_failed:>6} {raw_shape_str:>18} "
            f"{est.raw_size_gb:>8.2f} {est.estimated_processed_size_gb:>8.2f}"
        )

        for tile_name, errors in est.failed_tiles:
            err_str = "; ".join(errors)
            if len(err_str) > 80:
                err_str = err_str[:77] + "..."
            lines.append(f"  FAILED {tile_name}: {err_str}")

        total_raw += est.raw_size_gb
        total_proc += est.estimated_processed_size_gb
        total_tiles_ok += est.n_tiles_ok
        total_tiles_failed += est.n_tiles_failed

    lines.append("-" * 100)
    lines.append(
        f"{'TOTAL':<50} {'':>3} {'':>5} {total_tiles_ok:>6} {total_tiles_failed:>6} "
        f"{'':>18} {total_raw:>8.2f} {total_proc:>8.2f}"
    )

    if output_folder:
        has_space, free_gb = check_disk_capacity(output_folder, total_proc)
        status = "OK" if has_space else "INSUFFICIENT"
        lines.append(f"\nDisk free: {free_gb:.2f} GB at {output_folder} [{status}]")

    lines.append("")
    if any(est.channel_time_sizes and len(set(est.channel_time_sizes.values())) > 1 for est in estimates):
        lines.append("* = channels have different timepoint counts (shorter channels will be zero-padded)")

    lines.append("=" * 100)
    return "\n".join(lines)

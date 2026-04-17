from __future__ import annotations

import re
import json
import logging
from typing import Any, Optional

from pipeline.scanner import ROIScanResult

logger = logging.getLogger(__name__)


def build_prepared_entry(
    roi_scan: ROIScanResult,
    channel_mapping: dict[str, str],
    metadata: dict,
    channel_size: int,
    cube_size: int = 128,
    cluster: str | None = None,
) -> dict[str, Any]:
    """Build a ``prepared`` table insert dict.

    All scalar fields (software_version, output_folder, ...) are pulled directly
    from the metadata dict produced by submit_jobs.py.
    """
    if not metadata:
        raise ValueError(
            "metadata dict is empty; processing must run before ingestion."
        )

    ok_tiles = {k: v for k, v in roi_scan.tiles.items() if v.status == "ok"}

    n_timepoints = 0
    for tile in ok_tiles.values():
        if tile.time_size:
            n_timepoints = max(n_timepoints, max(tile.time_size.values()))

    server_folder = metadata.get("server_folder")
    output_folder = metadata.get("output_folder")

    entry = {
        "software_version": metadata.get("software_version"),
        "output_folder": output_folder,
        "server_folder": server_folder,
        "data_location": f"{server_folder}/{output_folder}".strip("/"),
        "elapsed_sec": metadata.get("elapsed_sec"),
        "cube_size": cube_size,
        "time_size": n_timepoints,
        "channel_size": channel_size,
        "channel_mapping": json.dumps(channel_mapping),
        "raw_roi_acquisition_id": roi_scan.roi_acquisition_id,
        "is_available": True,
    }
    if cluster:
        entry[f"exists_{cluster}"] = True
    return entry


def build_prepared_tiles_entries(
    roi_scan: ROIScanResult,
    metadata: dict,
    channel_size: int,
) -> list[dict[str, Any]]:
    """Build ``prepared_tiles`` insert dicts.

    Iterates ``metadata['training_images']`` keys (zarr channel patterns) so
    that ``tile_name`` matches what :func:`build_prepared_cubes_entries` produces.

    Processed dimensions (``n_z/n_y/n_x``) come from each tile's ``bbox``;
    raw dimensions come from the scan.  No fallback — raises if metadata is
    incomplete.
    """
    training_images = metadata.get("training_images")
    if not training_images:
        raise ValueError(
            "metadata has no 'training_images'; cannot build prepared_tiles. "
            "Processing must run before ingestion."
        )

    raw_shape = roi_scan.image_shape

    tiles: list[dict[str, Any]] = []
    for zarr_pattern, image_data in training_images.items():
        bbox = image_data.get("bbox")
        if bbox is None or len(bbox) < 6:
            raise ValueError(
                f"training_images[{zarr_pattern!r}] has no valid 'bbox'; "
                "cannot derive processed shape."
            )

        n_z = int(bbox[3] - bbox[0])
        n_y = int(bbox[4] - bbox[1])
        n_x = int(bbox[5] - bbox[2])

        chunk_names = image_data.get("chunk_names", {})
        timepoints: set[int] = set()
        for chunk_name in chunk_names:
            parts = re.split(r"[./]", chunk_name)
            if "c" in parts:
                parts.remove("c")
            if len(parts) >= 2:
                timepoints.add(int(parts[1]))
        n_timepoints = len(timepoints) if timepoints else 0

        processed_voxels = n_z * n_y * n_x
        processed_size_gb = round(
            n_timepoints * channel_size * processed_voxels * 2 / (1 << 30), 6
        )

        entry: dict[str, Any] = {
            "tile_name": zarr_pattern,
            "is_test_split": False,
            "time_size": n_timepoints,
            "n_timepoints": n_timepoints,
            "channel_size": channel_size,
            "n_z": n_z,
            "n_y": n_y,
            "n_x": n_x,
            "processed_size_gb": processed_size_gb,
        }

        if raw_shape:
            entry["raw_n_z"] = raw_shape[0]
            entry["raw_n_y"] = raw_shape[1]
            entry["raw_n_x"] = raw_shape[2]
            raw_voxels = raw_shape[0] * raw_shape[1] * raw_shape[2]
            entry["raw_size_gb"] = round(
                n_timepoints * channel_size * raw_voxels * 2 / (1 << 30), 6
            )

        tiles.append(entry)

    return tiles


def _extract_cdf(histogram: dict, percentile: float) -> Optional[int]:
    """Extract a CDF value from a histogram dict {percentile_str: value}."""
    key = str(percentile)
    if key in histogram:
        return int(histogram[key])
    for k, v in histogram.items():
        if abs(float(k) - percentile) < 0.01:
            return int(v)
    return None


def build_prepared_cubes_entries(
    metadata: dict,
    output_zarr_version: str = "zarr3",
) -> list[dict[str, Any]]:
    """Build ``prepared_cubes`` insert dicts from submit_jobs metadata.

    Per-chunk bbox values are required — raises on missing data.
    """
    cubes: list[dict[str, Any]] = []
    delimiters = r"[./]"

    training_images = metadata.get("training_images", {})
    for zarr_filename, image_data in training_images.items():
        tile_name = zarr_filename
        for chunk_name, chunk_meta in image_data.get("chunk_names").items():
            parts = re.split(delimiters, chunk_name)
            if "c" in parts:
                parts.remove("c")

            entry: dict[str, Any] = {
                "tile_name": tile_name,
                "chunk": int(parts[0]),
                "time": int(parts[1]),
                "z_start": chunk_meta["bbox"][0],
                "y_start": chunk_meta["bbox"][1],
                "x_start": chunk_meta["bbox"][2],
                "channel": int(parts[5]) if len(parts) > 5 else 0,
                "occupancy_ratio": chunk_meta.get("occ_ratio"),
            }

            histogram = chunk_meta.get("histogram", {})
            if histogram:
                for pct, col in [
                    (80.0, "cdf_80"),
                    (90.0, "cdf_90"),
                    (95.0, "cdf_95"),
                    (99.0, "cdf_99"),
                ]:
                    val = _extract_cdf(histogram, pct)
                    if val is not None:
                        entry[col] = val

            cubes.append(entry)

    return cubes

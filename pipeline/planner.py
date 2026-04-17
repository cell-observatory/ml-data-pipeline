from __future__ import annotations

import json
import uuid
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from pipeline.scanner import ROIScanResult
from pipeline.estimator import ROISizeEstimate

logger = logging.getLogger(__name__)


@dataclass
class ROIPlan:
    roi_acquisition_id: str
    roi_path: str
    channel_mapping: dict[str, str]
    tiff_token_mapping: dict[str, str]  # ch_idx -> tiff_token
    image_shape: Optional[tuple[int, int, int]]  # raw z, y, x
    n_timepoints: int
    time_size: dict[str, int]  # per-channel
    tiles: dict[str, dict]  # tile_name -> {files_per_channel: {ch_idx: [filenames]}}
    channel_size: int
    preprocessing: dict[str, Any] = field(default_factory=dict)  # csc/decon/dsr flags + params


@dataclass
class ProcessingPlan:
    run_id: str
    output_folder: str
    cube_shape: tuple[int, int, int]
    batch_size: int
    output_zarr_version: str
    rois: list[ROIPlan]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> ProcessingPlan:
        d = json.loads(s)
        d["cube_shape"] = tuple(d["cube_shape"])
        d["rois"] = [ROIPlan(**r) for r in d["rois"]]
        for roi in d["rois"]:
            if roi.image_shape:
                roi.image_shape = tuple(roi.image_shape)
        return cls(**d)

    def to_datasets_dict(self) -> dict[str, dict]:
        """Convert this plan into the datasets dict format expected by submit_jobs.py functions.

        The returned dict is keyed by roi_path and has the structure that
        submit_jobs.py functions expect: channelPatterns, preprocessing flags, etc.
        """
        datasets = {}
        for roi in self.rois:
            dataset = dict(roi.preprocessing)
            channel_patterns = []
            for ch_idx in sorted(roi.tiff_token_mapping.keys(), key=int):
                token = roi.tiff_token_mapping[ch_idx]
                parts = token.split(":")
                if len(parts) >= 2:
                    channel_patterns.append(f"{parts[0]}_{parts[1]}")
                else:
                    channel_patterns.append(token)
            dataset["channelPatterns"] = channel_patterns
            datasets[roi.roi_path] = dataset
        return datasets


def build_processing_plan(
    scan_results: list[ROIScanResult],
    size_estimates: list[ROISizeEstimate],
    channel_mappings: list[dict[str, str]],
    tiff_token_mappings: list[dict[str, str]],
    output_folder: str,
    cube_shape: tuple[int, int, int] = (128, 128, 128),
    batch_size: int = 16,
    output_zarr_version: str = "zarr3",
    preprocessing_configs: Optional[list[dict[str, Any]]] = None,
    run_id: Optional[str] = None,
) -> ProcessingPlan:
    """Build a ProcessingPlan from scan results, estimates, and mappings.

    preprocessing_configs is a per-ROI list of dicts (from PreprocessingProfile).
    If None, every ROI gets an empty preprocessing dict.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]

    if preprocessing_configs is None:
        preprocessing_configs = [{} for _ in scan_results]

    roi_plans = []
    for scan, est, ch_map, tok_map, preproc in zip(
        scan_results, size_estimates, channel_mappings, tiff_token_mappings,
        preprocessing_configs,
    ):
        ok_tiles = {}
        for tile_name, tile in scan.tiles.items():
            if tile.status == "ok":
                ok_tiles[tile_name] = {
                    "files_per_channel": tile.files_per_channel,
                }

        roi_plans.append(
            ROIPlan(
                roi_acquisition_id=scan.roi_acquisition_id,
                roi_path=scan.roi_path,
                channel_mapping=ch_map,
                tiff_token_mapping=tok_map,
                image_shape=scan.image_shape,
                n_timepoints=est.max_timepoints,
                time_size=est.channel_time_sizes,
                tiles=ok_tiles,
                channel_size=len(ch_map),
                preprocessing=preproc,
            )
        )

    return ProcessingPlan(
        run_id=run_id,
        output_folder=output_folder,
        cube_shape=cube_shape,
        batch_size=batch_size,
        output_zarr_version=output_zarr_version,
        rois=roi_plans,
    )

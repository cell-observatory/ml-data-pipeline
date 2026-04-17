from __future__ import annotations

import os
import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from tqdm import tqdm

logger = logging.getLogger(__name__)

TIFF_REGEX = re.compile(
    r"(?P<camera>Cam[A-Z])_(?P<ch>ch\d+)_"
    r".*?_(?P<wavelength>\d+)nm_"
    r".*?_(?P<msec_abs>\d+)msecAbs_"
    r"(?P<tile_x>-?\d+)x_(?P<tile_y>-?\d+)y_(?P<tile_z>-?\d+)z_"
    r"(?P<timepoint>\d+)t\.tif$"
)


@dataclass
class ParsedTiff:
    filename: str
    camera: str
    ch: str
    wavelength: str
    msec_abs: int
    tile_x: int
    tile_y: int
    tile_z: int
    timepoint: int


@dataclass
class TileScanResult:
    tile_name: str
    status: str  # "ok" | "failed"
    error_messages: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    time_size: dict[str, int] = field(default_factory=dict)
    n_timepoints: int = 0
    image_shape: Optional[tuple[int, int, int]] = None
    files_per_channel: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ROIScanResult:
    roi_acquisition_id: str
    roi_path: str
    tiles: dict[str, TileScanResult] = field(default_factory=dict)
    image_shape: Optional[tuple[int, int, int]] = None


def format_tile_name(x: int, y: int, z: int) -> str:
    return f"{x:03d}x_{y:03d}y_{z:03d}z"


def parse_tiff_filename(filename: str) -> Optional[ParsedTiff]:
    m = TIFF_REGEX.search(filename)
    if not m:
        return None
    return ParsedTiff(
        filename=filename,
        camera=m.group("camera"),
        ch=m.group("ch"),
        wavelength=m.group("wavelength"),
        msec_abs=int(m.group("msec_abs")),
        tile_x=int(m.group("tile_x")),
        tile_y=int(m.group("tile_y")),
        tile_z=int(m.group("tile_z")),
        timepoint=int(m.group("timepoint")),
    )


def _decompose_tiff_token(token: str) -> dict[str, str]:
    """Parse a DB tiff_token like 'CamB:ch0:560' into components."""
    parts = token.split(":")
    result = {}
    if len(parts) >= 1:
        result["camera"] = parts[0]
    if len(parts) >= 2:
        result["ch"] = parts[1]
    if len(parts) >= 3:
        result["wavelength"] = parts[2]
    return result


def _tiff_matches_token(parsed: ParsedTiff, token_parts: dict[str, str]) -> bool:
    for key, val in token_parts.items():
        if getattr(parsed, key, None) != val:
            return False
    return True


def scan_roi(
    roi_acquisition_id: str,
    roi_path: str,
    biological_channel_to_tiff_tokens: dict[str, str],
    channel_mapping: dict[str, str],
) -> ROIScanResult:
    """Scan an ROI directory and produce per-tile validation results.

    Parameters
    ----------
    roi_acquisition_id : str
        UUID of the ROI acquisition.
    roi_path : str
        Full filesystem path to the ROI directory containing TIFFs.
    biological_channel_to_tiff_tokens : dict
        Maps biological channel name -> tiff token string (e.g. "membrane" -> "CamB:ch0:560").
    channel_mapping : dict
        Maps output channel index (str) -> biological channel name
        (e.g. {"0": "membrane", "1": "nucleus"}).
    """
    result = ROIScanResult(roi_acquisition_id=roi_acquisition_id, roi_path=roi_path)
    roi_dir = Path(roi_path)
    if not roi_dir.is_dir():
        single_tile = TileScanResult(
            tile_name="000x_000y_000z",
            status="failed",
            error_messages=[f"ROI path does not exist: {roi_path}"],
        )
        result.tiles["000x_000y_000z"] = single_tile
        return result

    # Build token -> channel_index + token_parts lookup
    token_to_channel: dict[str, str] = {}  # tiff_token -> channel index
    token_parts_map: dict[str, dict[str, str]] = {}  # tiff_token -> decomposed parts
    for ch_idx, bio_name in channel_mapping.items():
        tiff_token = biological_channel_to_tiff_tokens.get(bio_name)
        if not tiff_token:
            single_tile = TileScanResult(
                tile_name="000x_000y_000z",
                status="failed",
                error_messages=[
                    f"No tiff_token found for biological channel '{bio_name}' "
                    f"in biological_channel_to_tiff_tokens"
                ],
            )
            result.tiles["000x_000y_000z"] = single_tile
            return result
        token_to_channel[tiff_token] = ch_idx
        token_parts_map[tiff_token] = _decompose_tiff_token(tiff_token)

    # Step 1: Parse all TIFFs
    parsed_files: list[ParsedTiff] = []
    for fname in tqdm(
        os.listdir(roi_dir),
        desc=f"Parsing TIFFs {roi_acquisition_id}",
        leave=False,
    ):
        if not fname.endswith(".tif"):
            continue
        parsed = parse_tiff_filename(fname)
        if parsed:
            parsed_files.append(parsed)

    if not parsed_files:
        single_tile = TileScanResult(
            tile_name="000x_000y_000z",
            status="failed",
            error_messages=["No parseable TIFF files found in ROI directory"],
        )
        result.tiles["000x_000y_000z"] = single_tile
        return result

    # Step 2: Match files to DB tokens -> assign channel index
    # file -> (channel_index, token)
    file_channel_map: dict[str, tuple[str, str]] = {}
    unmatched: list[str] = []
    for pf in parsed_files:
        matched = False
        for token, parts in token_parts_map.items():
            if _tiff_matches_token(pf, parts):
                file_channel_map[pf.filename] = (token_to_channel[token], token)
                matched = True
                break
        if not matched:
            unmatched.append(pf.filename)

    matched_files = [pf for pf in parsed_files if pf.filename in file_channel_map]
    if not matched_files:
        single_tile = TileScanResult(
            tile_name="000x_000y_000z",
            status="failed",
            error_messages=[
                "No TIFF files matched any DB tiff_token. "
                f"Tokens: {list(token_parts_map.keys())}. "
                f"Sample files: {[pf.filename for pf in parsed_files[:3]]}"
            ],
        )
        result.tiles["000x_000y_000z"] = single_tile
        return result

    # Step 3: Group by tile
    tiles: dict[str, list[tuple[ParsedTiff, str]]] = {}  # tile_name -> [(parsed, ch_idx)]
    for pf in matched_files:
        tile_name = format_tile_name(pf.tile_x, pf.tile_y, pf.tile_z)
        ch_idx = file_channel_map[pf.filename][0]
        tiles.setdefault(tile_name, []).append((pf, ch_idx))

    # Load one sample file to get image shape
    sample_file = matched_files[0]
    try:
        import cpptiff
        image_shape = tuple(cpptiff.get_image_shape(str(roi_dir / sample_file.filename)))
        result.image_shape = image_shape
    except Exception as e:
        logger.warning("Could not read image shape from %s: %s", sample_file.filename, e)
        image_shape = None

    expected_channels = set(channel_mapping.keys())

    for tile_name, tile_files in tqdm(
        sorted(tiles.items()),
        desc=f"Validating tiles {roi_acquisition_id}",
        leave=False,
    ):
        tile_result = TileScanResult(tile_name=tile_name, status="failed")

        # Group by (channel_index, timepoint)
        channel_groups: dict[str, list[ParsedTiff]] = {}
        for pf, ch_idx in tile_files:
            channel_groups.setdefault(ch_idx, []).append(pf)

        # Check all expected channels present
        present_channels = set(channel_groups.keys())
        missing = expected_channels - present_channels
        if missing:
            tile_result.status = "failed"
            tile_result.error_messages.append(
                f"Missing channels: {sorted(missing)}. "
                f"Present: {sorted(present_channels)}"
            )
            result.tiles[tile_name] = tile_result
            continue

        tile_time_size: dict[str, int] = {}
        tile_files_per_channel: dict[str, list[str]] = {}
        channel_failed = False

        for ch_idx in sorted(expected_channels):
            ch_files = channel_groups[ch_idx]

            # Step 3a: Deduplicate — group by timepoint, keep most recent msecAbs
            tp_groups: dict[int, list[ParsedTiff]] = {}
            for pf in ch_files:
                tp_groups.setdefault(pf.timepoint, []).append(pf)

            deduped: dict[int, ParsedTiff] = {}
            for tp, files in tp_groups.items():
                if len(files) > 1:
                    files.sort(key=lambda f: f.msec_abs, reverse=True)
                    tile_result.warnings.append(
                        f"ch{ch_idx} tp{tp}: {len(files)} duplicates, "
                        f"kept msecAbs={files[0].msec_abs}"
                    )
                deduped[tp] = files[0]

            # Step 3b: Verify contiguous from 0..N
            sorted_tps = sorted(deduped.keys())
            if not sorted_tps:
                tile_result.status = "failed"
                tile_result.error_messages.append(f"ch{ch_idx}: no valid timepoints")
                channel_failed = True
                break

            expected_tps = list(range(sorted_tps[0], sorted_tps[-1] + 1))
            if sorted_tps != expected_tps:
                gaps = set(expected_tps) - set(sorted_tps)
                tile_result.status = "failed"
                tile_result.error_messages.append(
                    f"ch{ch_idx}: non-contiguous timepoints. "
                    f"Range {sorted_tps[0]}-{sorted_tps[-1]}, "
                    f"missing: {sorted(gaps)[:10]}{'...' if len(gaps) > 10 else ''}"
                )
                channel_failed = True
                break

            if sorted_tps[0] != 0:
                tile_result.status = "failed"
                tile_result.error_messages.append(
                    f"ch{ch_idx}: timepoints don't start at 0 (starts at {sorted_tps[0]})"
                )
                channel_failed = True
                break

            # Step 3c: Verify msecAbs strictly increasing
            # TODO: is this correct?
            ordered = [deduped[tp] for tp in sorted_tps]
            for i in range(1, len(ordered)):
                if ordered[i].msec_abs <= ordered[i - 1].msec_abs:
                    tile_result.status = "failed"
                    tile_result.error_messages.append(
                        f"ch{ch_idx}: msecAbs not strictly increasing at "
                        f"tp{sorted_tps[i]} ({ordered[i].msec_abs} <= {ordered[i-1].msec_abs})"
                    )
                    channel_failed = True
                    break
            if channel_failed:
                break

            tile_time_size[ch_idx] = len(sorted_tps)
            tile_files_per_channel[ch_idx] = [deduped[tp].filename for tp in sorted_tps]

        if channel_failed:
            if tile_result.status != "failed":
                tile_result.status = "failed"
            result.tiles[tile_name] = tile_result
            continue

        # Step 4: Build per-channel time_size dict
        tile_result.time_size = tile_time_size
        tile_result.n_timepoints = max(tile_time_size.values()) if tile_time_size else 0
        tile_result.image_shape = image_shape
        tile_result.files_per_channel = tile_files_per_channel
        tile_result.status = "ok"

        if len(set(tile_time_size.values())) > 1:
            tile_result.warnings.append(
                f"Channels have different timepoint counts: {tile_time_size}. "
                f"Zarr will use max={tile_result.n_timepoints}, "
                f"shorter channels zero-padded."
            )

        result.tiles[tile_name] = tile_result

    return result


def scan_result_to_log_entries(
    scan_result: ROIScanResult,
    pipeline_run_id: str = "",
) -> list[dict]:
    """Convert an ROIScanResult into scan log entries for persistence."""
    entries = []
    for tile_name, tile in scan_result.tiles.items():
        scan_metadata = {
            "time_size": tile.time_size,
            "n_timepoints": tile.n_timepoints,
        }
        if tile.image_shape:
            scan_metadata["image_shape"] = list(tile.image_shape)
        if tile.warnings:
            scan_metadata["warnings"] = tile.warnings

        entry = {
            "roi_acquisition_id": scan_result.roi_acquisition_id,
            "tile_name": tile_name,
            "status": tile.status,
            "error_messages": tile.error_messages if tile.error_messages else None,
            "scan_metadata": scan_metadata,
        }
        if pipeline_run_id:
            entry["pipeline_run_id"] = pipeline_run_id
        entries.append(entry)
    return entries

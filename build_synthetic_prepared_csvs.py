"""Convert synthetic-data sidecar CSVs into the four CSVs that
``add_synthetic_metadata_to_db.py`` consumes (prepared.csv,
prepared_tiles.csv, prepared_cubes.csv, prepared_annotations_3d.csv).

Input layout (one ROI):

    <root>/<DATE_dataset>/<fish>/<roi>/
        <tile>.zarr/
        <tile>_mask_bbox.csv
        <tile>_mem_occratio.csv
        <tile>_cytosol_occratio.csv
        <tile>_cytosol_fp_expressions.csv     # not consumed here
        <tile>_staging_annotations_3d.csv     # consumed: per-instance bboxes
                                              # rolled up into annotations_3d

Plus a single ``channel_mapping.json`` shared across the dataset, of the
shape produced by ``generate_channel_mapping.py``::

    {"0": "<membrane-fp>-membrane",
     "1": "<fp1>-cytosol",
     "2": "<fp2>-cytosol",
     ...,
     "<N>": "instance-mask"}

Cytosol has multiple ``fluorescent_protein`` rows per cube; each is exposed
as its own logical channel in ``prepared_cubes`` (one row per channel per
cube), so the channel index lines up with channel_mapping.json.

The final entry MUST be the dense instance-id labelmap (role=``mask``) at
the highest index. The cell_observatory_platform dataloader slices
``c = slice(0, channel_size)`` from the zarr and then clones
``inputs[..., -1]`` as the integer labelmap, so:

  - ``channel_size`` in ``prepared`` / ``prepared_tiles`` MUST equal
    ``zarr.shape[-1]`` (validated per-tile against the zarr)
  - the mask channel is intentionally excluded from ``prepared_cubes``
    (it has no meaningful intensity histogram / occupancy_ratio)
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd
import zarr


DEFAULT_CUBE_SIZE = 128
SOFTWARE_VERSION = "synthetic_data_iteration_2"

MEMBRANE_ROLE = "membrane"
CYTOSOL_ROLE = "cytosol"
MASK_ROLE = "mask"
VALID_ROLES = (MEMBRANE_ROLE, CYTOSOL_ROLE, MASK_ROLE)

# Roles whose cubes carry per-channel intensity stats (occ_ratio + cdf).
# The mask channel is an integer instance-id labelmap, so it has no
# meaningful histogram and is intentionally skipped in prepared_cubes.
INTENSITY_ROLES = (MEMBRANE_ROLE, CYTOSOL_ROLE)


def extract_cdf(histogram: dict, percentile: float) -> int | None:
    """Mirror of pipeline/ingest.py:_extract_cdf, tolerant of float keys."""
    key = str(percentile)
    if key in histogram:
        return int(histogram[key])
    for k, v in histogram.items():
        if abs(float(k) - percentile) < 0.01:
            return int(v)
    return None


def parse_histogram(cell) -> dict:
    if not isinstance(cell, str):
        return {}
    return json.loads(cell)


def load_channel_mapping(path: str) -> list[tuple[int, str, str, str]]:
    """Parse channel_mapping.json into an ordered list of channels.

    Returns: [(channel_index, fp_name, role, full_name), ...] sorted by
    channel_index. Validates:
      - indices are contiguous from 0 to N-1 (no gaps, no duplicates)
      - each value is in '<fp>-<role>' format with role in VALID_ROLES
      - exactly 1 membrane channel
      - at least 1 cytosol channel
      - exactly 1 mask channel, AND it's the highest index (last entry)
    """
    with open(path) as f:
        raw = json.load(f)

    channels: list[tuple[int, str, str, str]] = []
    for k in sorted(raw.keys(), key=int):
        full_name = raw[k]
        if "-" not in full_name:
            raise ValueError(
                f"{path}: channel {k!r} value {full_name!r} is not in "
                f"'<fp>-<role>' format"
            )
        fp_name, role = full_name.rsplit("-", 1)
        if role not in VALID_ROLES:
            raise ValueError(
                f"{path}: channel {k!r} role {role!r} must be one of "
                f"{VALID_ROLES}"
            )
        channels.append((int(k), fp_name, role, full_name))

    indices = [idx for idx, _, _, _ in channels]
    if indices != list(range(len(channels))):
        raise ValueError(
            f"{path}: channel indices {indices} are not contiguous from 0 "
            f"to {len(channels) - 1}"
        )

    n_mem = sum(1 for _, _, role, _ in channels if role == MEMBRANE_ROLE)
    n_cyt = sum(1 for _, _, role, _ in channels if role == CYTOSOL_ROLE)
    n_mask = sum(1 for _, _, role, _ in channels if role == MASK_ROLE)
    if n_mem != 1:
        raise ValueError(
            f"{path}: expected exactly 1 membrane channel, got {n_mem}"
        )
    if n_cyt < 1:
        raise ValueError(
            f"{path}: expected at least 1 cytosol channel, got 0"
        )
    if n_mask != 1:
        raise ValueError(
            f"{path}: expected exactly 1 mask channel (the dense "
            f"instance-id labelmap), got {n_mask}"
        )
    last_idx, _, last_role, _ = channels[-1]
    if last_role != MASK_ROLE:
        raise ValueError(
            f"{path}: mask channel must be the highest index, but the "
            f"last channel ({last_idx}) has role {last_role!r}. "
            f"cell_observatory_platform's dataloader clones "
            f"inputs[..., -1] as the labelmap, so any other channel at "
            f"the last index would be silently misinterpreted as masks."
        )
    return channels


def _validate_channel_mapping_against_zarr(
    channels: list[tuple[int, str, str, str]],
    zarr_channel_dim: int,
    tile_path: str,
) -> None:
    """Ensure channel_mapping.json describes every channel stored in the zarr.

    cell_observatory_platform's dataloader slices
    ``c = slice(0, channel_size)`` from the zarr and then strips the last
    channel as the mask. ``channel_size`` therefore MUST equal
    ``zarr.shape[-1]``; if the mapping under-counts, the wrong channel
    gets cloned as the mask labelmap and targets become noise.
    """
    n_mapping = len(channels)
    if n_mapping != zarr_channel_dim:
        full_names = [full for *_, full in channels]
        raise ValueError(
            f"{tile_path}: channel_mapping has {n_mapping} entries "
            f"({full_names}) but zarr stores {zarr_channel_dim} channels. "
            f"They must match -- the mapping covers every channel "
            f"including the dense instance-id labelmap as the last entry."
        )


def _zarr_shape_5d(tile_path: str) -> tuple[int, int, int, int, int]:
    """Read shape of a (T, Z, Y, X, C) zarr written by convert_files.py."""
    z = zarr.open(tile_path, mode="r")
    shape = tuple(int(s) for s in z.shape)
    if len(shape) != 5:
        raise ValueError(
            f"{tile_path}: expected 5D (T,Z,Y,X,C) zarr, got shape {shape}"
        )
    return shape  # type: ignore[return-value]


def _load_membrane_lookup(path: str) -> pd.DataFrame:
    """Indexed mem_occratio.csv -> rows keyed by (timepoint, cube_id)."""
    df = pd.read_csv(path)
    df["histogram"] = df["histogram"].map(parse_histogram)
    return df.set_index(["timepoint", "cube_id"])


# Subset of <tile>_staging_annotations_3d.csv columns we forward to the DB.
# The platform's aggregate_prepared_tile_annotation_agg_1() function reads
# (z_start, y_start, x_start, z_end, y_end, x_end) -> bbox_zyxzyx and
# emits jsonb_build_object('local_segmentation_id', a.local_segmentation_id,
# 'cell_type_id', a.cell_type_id, 'bbox_zyxzyx', ...). It also requires
# is_consensus = true; we override the staging value (False) to True since
# synthetic annotations are ground truth by construction.
_ANNOTATION_FORWARDED_COLS = (
    "local_segmentation_id",
    "segmentation_conf",
    "timepoint",
    "x_start", "y_start", "z_start",
    "x_end",   "y_end",   "z_end",
    "cell_type_id",
    "annotator_id", "proofreader_id", "model_id",
    "exists_prfs", "exists_aws", "exists_oak",
    "exists_abc",  "exists_nersc", "exists_gcp",
)


def _load_staging_annotations(path: str, tile_name: str) -> pd.DataFrame:
    """Load <tile>_staging_annotations_3d.csv and shape it for our output.

    Drops the staging file's per-tile numeric ``id`` and ``tile_id`` (the
    DB join is on text ``tile_id = tile_name``, and ``id`` will be
    renumbered globally per ROI in ``walk``). ``is_consensus`` is forced
    to True; the staging file has it False but the platform agg ignores
    rows with is_consensus=false, so synthetic data would never roll up
    without the override.
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=_ANNOTATION_FORWARDED_COLS)
    df = pd.read_csv(path)
    missing = [c for c in _ANNOTATION_FORWARDED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{tile_name}: {path} is missing expected annotation columns: "
            f"{missing}"
        )
    return df[list(_ANNOTATION_FORWARDED_COLS)].copy()


def _load_cytosol_lookup(path: str,
                         allowed_fps: set[str],
                         tile_name: str) -> dict[str, pd.DataFrame]:
    """cytosol_occratio.csv split into one indexed DataFrame per FP.

    The set of cytosol FPs in the CSV must match ``allowed_fps`` exactly
    -- a missing FP means the channel_mapping promised data we don't
    have, and an extra FP means the underlying zarr probably has a
    cytosol channel that nobody is tracking.
    """
    df = pd.read_csv(path)
    df["histogram"] = df["histogram"].map(parse_histogram)

    csv_fps = set(df["fluorescent_protein"].unique())
    missing = allowed_fps - csv_fps
    if missing:
        raise KeyError(
            f"{tile_name}: cytosol FPs {sorted(missing)} from channel_mapping "
            f"are not present in {path} (CSV has {sorted(csv_fps)})"
        )
    extra = csv_fps - allowed_fps
    if extra:
        raise ValueError(
            f"{tile_name}: cytosol_occratio.csv has FPs {sorted(extra)} "
            f"not present in channel_mapping (mapping cytosol FPs: "
            f"{sorted(allowed_fps)}). Either add them to channel_mapping "
            f"(one channel per FP) or drop them from the CSV."
        )

    return {
        fp: group.set_index(["timepoint", "cube_id"])
        for fp, group in df.groupby("fluorescent_protein")
    }


def build_for_tile(roi_dir: str,
                   tile_name: str,
                   prepared_id: int,
                   channels: list[tuple[int, str, str, str]],
                   ) -> tuple[dict, list[dict], list[dict]]:
    """Build (tile_row, [cube_rows], [annotation_rows]) for one *.zarr tile.

    - cube_rows: one row per (cube, timepoint, channel) combination for
      every channel with role in INTENSITY_ROLES (membrane + cytosol).
      The mask channel is intentionally skipped -- it's an integer
      labelmap with no histogram/occupancy_ratio.
    - annotation_rows: one row per per-instance bounding box from the
      tile's staging_annotations_3d.csv, ready to be COPYed into
      public.annotations_3d after the prepared row gets a real DB id.
      ``id`` is omitted here; ``walk`` renumbers it globally per ROI.
    """
    stem = tile_name[: -len(".zarr")]

    bbox_path = os.path.join(roi_dir, f"{stem}_mask_bbox.csv")
    mem_path = os.path.join(roi_dir, f"{stem}_mem_occratio.csv")
    cyt_path = os.path.join(roi_dir, f"{stem}_cytosol_occratio.csv")
    anno_path = os.path.join(roi_dir, f"{stem}_staging_annotations_3d.csv")

    for required in (bbox_path, mem_path, cyt_path):
        if not os.path.exists(required):
            raise FileNotFoundError(
                f"{tile_name}: expected sidecar CSV missing: {required}"
            )

    bboxes = pd.read_csv(bbox_path)
    mem_lookup = _load_membrane_lookup(mem_path)
    cytosol_fps_needed = {fp for _, fp, role, _ in channels
                          if role == CYTOSOL_ROLE}
    cyt_lookup = _load_cytosol_lookup(cyt_path, cytosol_fps_needed, tile_name)

    tile_path = os.path.join(roi_dir, tile_name)
    n_t, n_z, n_y, n_x, n_ch_zarr = _zarr_shape_5d(tile_path)
    _validate_channel_mapping_against_zarr(channels, n_ch_zarr, tile_path)

    cube_rows: list[dict] = []
    for _, b in bboxes.iterrows():
        key = (b.timepoint, b.cube_id)
        for ch_idx, fp_name, role, _full_name in channels:
            if role not in INTENSITY_ROLES:
                continue

            lookup = mem_lookup if role == MEMBRANE_ROLE else cyt_lookup[fp_name]

            if key in lookup.index:
                row = lookup.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                occ_ratio = float(row["occupancy_ratio"])
                histogram = row["histogram"]
            else:
                occ_ratio = 0.0
                histogram = {}

            cube_rows.append({
                "prepared_id": prepared_id,
                "tile_name": tile_name,
                "chunk": int(b.cube_id),
                "time": int(b.timepoint),
                "z_start": int(b.cropped_z_start),
                "y_start": int(b.cropped_y_start),
                "x_start": int(b.cropped_x_start),
                "channel": ch_idx,
                "occupancy_ratio": occ_ratio,
                "cdf_80": extract_cdf(histogram, 80.0),
                "cdf_90": extract_cdf(histogram, 90.0),
                "cdf_95": extract_cdf(histogram, 95.0),
                "cdf_99": extract_cdf(histogram, 99.0),
            })

    tile_row = {
        "prepared_id": prepared_id,
        "tile_name": tile_name,
        "is_test_split": False,
        "channel_size": len(channels),
        "time_size": n_t,
        "n_timepoints": n_t,
        "n_z": n_z,
        "n_y": n_y,
        "n_x": n_x,
    }

    annotations_df = _load_staging_annotations(anno_path, tile_name)
    annotation_rows: list[dict] = []
    for _, ann in annotations_df.iterrows():
        rec: dict = {
            "prepared_id": prepared_id,    # local id; remapped at ingest
            "tile_id": tile_name,          # text join key against ptv.tile_name
            "is_consensus": True,          # synthetic == ground truth
        }
        for col in _ANNOTATION_FORWARDED_COLS:
            val = ann[col]
            # Drop NaNs so nullable columns end up as SQL NULL rather than
            # the string "nan". psycopg's adapter and pandas-to-csv both
            # cooperate with empty strings here.
            if pd.isna(val):
                continue
            rec[col] = val
        annotation_rows.append(rec)

    return tile_row, cube_rows, annotation_rows


def walk(root: str,
         server_folder: str,
         channels: list[tuple[int, str, str, str]],
         skip_empty_rois: bool = True,
         ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Walk <root>/<DATE>/<fish>/<roi>/ and build four DataFrames.

    Returns ``(prepared, prepared_tiles, prepared_cubes, annotations_3d)``.
    Annotation rows omit ``id``: ``annotations_3d.id`` is a
    ``GENERATED ALWAYS`` identity in the DB; the ingester lets the
    server assign it (PK uniqueness is still ``(id, roi_id)`` once
    ``roi_id`` is stamped at insert time).
    """
    server_folder_clean = server_folder.rstrip("/")
    channel_mapping_dict = {str(idx): full
                            for idx, _fp, _role, full in channels}
    channel_size = len(channels)

    prepared_rows: list[dict] = []
    tile_rows: list[dict] = []
    cube_rows: list[dict] = []
    annotation_rows: list[dict] = []

    next_id = 0
    for date_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(date_dir):
            continue
        for fish_dir in sorted(glob.glob(os.path.join(date_dir, "*"))):
            if not os.path.isdir(fish_dir):
                continue
            for roi_dir in sorted(glob.glob(os.path.join(fish_dir, "*"))):
                if not os.path.isdir(roi_dir):
                    continue
                tiles = sorted(glob.glob(os.path.join(roi_dir, "*.zarr")))
                if not tiles:
                    if skip_empty_rois:
                        continue
                    raise FileNotFoundError(f"{roi_dir}: no *.zarr tiles found")

                rel_output = os.path.relpath(roi_dir, server_folder_clean)
                data_location = f"{server_folder_clean}/{rel_output}".strip("/")

                prepared_id = next_id
                next_id += 1

                this_tile_rows: list[dict] = []
                this_cube_rows: list[dict] = []
                this_anno_rows: list[dict] = []
                t_size = 0
                for tile_path in tiles:
                    tname = os.path.basename(tile_path)
                    trow, crows, arows = build_for_tile(
                        roi_dir, tname, prepared_id, channels,
                    )
                    this_tile_rows.append(trow)
                    this_cube_rows.extend(crows)
                    this_anno_rows.extend(arows)
                    t_size = max(t_size, trow["time_size"])

                prepared_rows.append({
                    "id": prepared_id,
                    "software_version": SOFTWARE_VERSION,
                    "output_folder": rel_output,
                    "server_folder": server_folder_clean,
                    "data_location": data_location,
                    "cube_size": DEFAULT_CUBE_SIZE,
                    "time_size": t_size,
                    "channel_size": channel_size,
                    "is_synthetic": True,
                    "channel_mapping": json.dumps(channel_mapping_dict),
                })
                tile_rows.extend(this_tile_rows)
                cube_rows.extend(this_cube_rows)
                annotation_rows.extend(this_anno_rows)

    return (
        pd.DataFrame(prepared_rows),
        pd.DataFrame(tile_rows),
        pd.DataFrame(cube_rows),
        pd.DataFrame(annotation_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", required=True,
        help="Synthetic data root, e.g. /clusterfs/vast/forsynthetic/"
             "benchmark_tests/iteration2_martin/synthetic_data_iteration_2",
    )
    ap.add_argument(
        "--server-folder", required=True,
        help="Prefix to record as prepared.server_folder. Usually identical "
             "to --root. Trailing slash will be stripped.",
    )
    ap.add_argument(
        "--channel-mapping", required=True,
        help="Path to channel_mapping.json (see generate_channel_mapping.py). "
             "Future synthetic-data iterations will emit this alongside the "
             "data automatically.",
    )
    ap.add_argument(
        "--out-dir", required=True,
        help="Directory where prepared.csv, prepared_tiles.csv, "
             "prepared_cubes.csv, and prepared_annotations_3d.csv will "
             "be written.",
    )
    args = ap.parse_args()

    channels = load_channel_mapping(args.channel_mapping)
    print(f"Loaded {len(channels)} channels from {args.channel_mapping}: "
          f"{[full for _, _, _, full in channels]}")

    prepared_df, tiles_df, cubes_df, anno_df = walk(
        args.root, args.server_folder, channels=channels,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    prepared_path = os.path.join(args.out_dir, "prepared.csv")
    tiles_path = os.path.join(args.out_dir, "prepared_tiles.csv")
    cubes_path = os.path.join(args.out_dir, "prepared_cubes.csv")
    anno_path = os.path.join(args.out_dir, "prepared_annotations_3d.csv")
    prepared_df.to_csv(prepared_path, index=False)
    tiles_df.to_csv(tiles_path, index=False)
    cubes_df.to_csv(cubes_path, index=False)
    anno_df.to_csv(anno_path, index=False)

    print(
        f"Wrote {len(prepared_df)} prepared / {len(tiles_df)} tiles / "
        f"{len(cubes_df)} cubes / {len(anno_df)} annotations to {args.out_dir}"
    )


if __name__ == "__main__":
    main()

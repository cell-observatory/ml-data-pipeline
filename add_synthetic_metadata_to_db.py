"""Insert synthetic-data CSVs into the acquisition DB.

Reads three CSVs produced by ``build_synthetic_prepared_csvs.py``:

- ``prepared.csv``       (one row per ROI)
- ``prepared_tiles.csv`` (rows joined to prepared via ``prepared_id``)
- ``prepared_cubes.csv`` (rows joined to prepared via ``prepared_id``)

and inserts them into ``prepared`` / ``prepared_tiles`` / ``prepared_cubes``,
then refreshes the aggregate cache artifacts. Targets either a local
Postgres sandbox (``--db local``, via psycopg) or Supabase REST
(``--db staging`` / ``--db prod``), reusing the same backends as
``run_pipeline.py`` (``pipeline.store.LocalPostgresStore`` /
``SupabaseStore``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.db_client import PipelineDBClient  # noqa: E402
from pipeline.store import create_store  # noqa: E402


CLUSTER_CHOICES = ("prfs", "aws", "oak", "abc", "nersc", "gcp")
DB_CHOICES = ("local", "staging", "prod")

_INT_COLS = {
    "prepared": ("time_size", "channel_size", "cube_size"),
    "prepared_tiles": (
        "channel_size", "time_size", "n_timepoints",
        "n_z", "n_y", "n_x", "raw_n_z", "raw_n_y", "raw_n_x",
    ),
    "prepared_cubes": (
        "chunk", "time", "channel",
        "z_start", "y_start", "x_start",
        "cdf_80", "cdf_90", "cdf_95", "cdf_99",
    ),
    "annotations_3d": (
        "local_segmentation_id", "timepoint",
        "x_start", "y_start", "z_start",
        "x_end",   "y_end",   "z_end",
        "cell_type_id", "annotator_id", "proofreader_id", "model_id",
    ),
}
_FLOAT_COLS = {
    "prepared": (),
    "prepared_tiles": ("raw_size_gb", "processed_size_gb"),
    "prepared_cubes": ("occupancy_ratio",),
    "annotations_3d": ("segmentation_conf",),
}
_BOOL_COLS = {
    "prepared": ("is_synthetic", "is_available"),
    "prepared_tiles": ("is_test_split",),
    "prepared_cubes": (),
    "annotations_3d": (
        "is_consensus",
        "exists_prfs", "exists_aws", "exists_oak",
        "exists_abc",  "exists_nersc", "exists_gcp",
    ),
}


def _row_to_dict(row, table: str) -> dict:
    """Convert a pandas itertuples row into an insert dict.

    - drops NaN values (they collide with NOT NULL columns and aren't really
      "I want to write NULL" — they're "this CSV doesn't have this column")
    - casts numeric / bool columns to their proper Python type so psycopg
      doesn't bind them as numpy scalars
    """
    out: dict = {}
    int_cols = _INT_COLS.get(table, ())
    float_cols = _FLOAT_COLS.get(table, ())
    bool_cols = _BOOL_COLS.get(table, ())
    for col in row._fields:
        val = getattr(row, col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if col in int_cols:
            val = int(val)
        elif col in float_cols:
            val = float(val)
        elif col in bool_cols:
            val = bool(val)
        out[col] = val
    return out


def _build_prepared_dict(row, cluster: str | None) -> dict:
    d = _row_to_dict(row, "prepared")
    # CSV-local id is only used to join to the per-ROI tile/cube rows; the
    # DB assigns its own SERIAL id and the store hands it back.
    d.pop("id", None)
    d.setdefault("elapsed_sec", 0)
    d.setdefault("is_available", True)
    if cluster:
        d[f"exists_{cluster}"] = True
    if "channel_mapping" in d and isinstance(d["channel_mapping"], str):
        d["channel_mapping"] = json.loads(d["channel_mapping"])
    return d


def _build_tile_dict(row) -> dict:
    d = _row_to_dict(row, "prepared_tiles")
    # store.ingest_prepared_roi rewrites prepared_id from the returned id.
    d.pop("prepared_id", None)
    return d


def _build_cube_dict(row) -> dict:
    d = _row_to_dict(row, "prepared_cubes")
    d.pop("prepared_id", None)
    return d


def _build_annotation_dict(row) -> dict:
    d = _row_to_dict(row, "annotations_3d")
    # store.ingest_annotations_3d sets roi_id from the real prepared_id;
    # the CSV-local prepared_id is only kept to join annotation rows to
    # their parent ROI here.
    d.pop("prepared_id", None)
    d.pop("roi_id", None)
    d.pop("id", None)  # optional CSV column; DB is GENERATED ALWAYS
    return d


def add_synthetic_metadata_to_db(
    metadata_file: str,
    db: str = "local",
    dotenv: str | None = None,
    cluster: str | None = None,
    supabase_url: str | None = None,
    supabase_key: str | None = None,
) -> tuple[list[int], list]:
    """Ingest synthetic CSVs; return ``(inserted_prepared_ids, failed_local_ids)``."""
    folder = os.path.dirname(metadata_file)
    # TODO: Pipeline this at the ROI level instead of requiring the full csv file before running
    df_prepared = pd.read_csv(metadata_file)
    df_tiles = pd.read_csv(os.path.join(folder, "prepared_tiles.csv"))
    df_cubes = pd.read_csv(os.path.join(folder, "prepared_cubes.csv"))
    anno_path = os.path.join(folder, "prepared_annotations_3d.csv")
    if os.path.exists(anno_path):
        df_anno = pd.read_csv(anno_path)
    else:
        # Backwards compat with CSV directories built before annotations
        # were emitted; ingest still runs but the annotation aggs stay empty.
        print(
            f"WARN: {anno_path} not found; annotations_3d will not be "
            f"populated and prepared_tile_annotation_agg_1 will stay empty "
            f"for these prepared_ids.",
            file=sys.stderr,
        )
        df_anno = pd.DataFrame(columns=["prepared_id"])

    db_client = PipelineDBClient(
        mode=db,
        dotenv_path=dotenv,
        supabase_url=supabase_url,
        supabase_key=supabase_key,
    )
    store = create_store("db", db_client=db_client)

    inserted_ids: list[int] = []
    failed_local: list = []
    total_annotations_inserted = 0

    for prepared_row in df_prepared.itertuples(index=False):
        local_id = getattr(prepared_row, "id", None)
        prepared_dict = _build_prepared_dict(prepared_row, cluster=cluster)
        tile_rows = df_tiles[df_tiles["prepared_id"] == local_id]
        cube_rows = df_cubes[df_cubes["prepared_id"] == local_id]
        anno_rows = df_anno[df_anno["prepared_id"] == local_id] \
            if "prepared_id" in df_anno.columns else df_anno.iloc[0:0]
        tile_dicts = [_build_tile_dict(r) for r in tile_rows.itertuples(index=False)]
        cube_dicts = [_build_cube_dict(r) for r in cube_rows.itertuples(index=False)]
        anno_dicts = [
            _build_annotation_dict(r) for r in anno_rows.itertuples(index=False)
        ]

        try:
            # Single transaction: prepared + tiles + cubes + annotations
            # all rollback together if anything fails. The store also
            # post-validates COUNT(*) FROM annotations_3d WHERE roi_id =
            # <new_id> equals len(anno_dicts) before COMMIT, so we can't
            # ship a half-ingested ROI.
            prepared_id = store.ingest_prepared_roi(
                prepared_dict, tile_dicts, cube_dicts,
                annotations=anno_dicts,
            )
            n_anno = len(anno_dicts)
            total_annotations_inserted += n_anno
            inserted_ids.append(prepared_id)
            print(
                f"Ingested local_id={local_id} -> prepared_id={prepared_id} "
                f"({len(tile_dicts)} tiles, {len(cube_dicts)} cubes, "
                f"{n_anno} annotations)"
            )
        except Exception as exc:
            failed_local.append(local_id)
            print(f"FAILED local_id={local_id}: {exc}", file=sys.stderr)

    if total_annotations_inserted:
        print(
            f"Inserted {total_annotations_inserted} rows into annotations_3d "
            f"across {len(inserted_ids)} ROIs"
        )

    if inserted_ids:
        print(
            f"Refreshing cache artifacts for {len(inserted_ids)} prepared_ids..."
        )
        failed_refresh = store.refresh_cache_artifacts(inserted_ids)
        if failed_refresh:
            print(
                f"WARN: cache refresh failed for prepared_ids {failed_refresh}",
                file=sys.stderr,
            )

    if failed_local:
        print(
            f"WARN: {len(failed_local)} ROIs failed to ingest: {failed_local}",
            file=sys.stderr,
        )

    return inserted_ids, failed_local


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metadata-file", required=True,
        help="Full path to prepared.csv. The two sibling CSVs "
             "(prepared_tiles.csv, prepared_cubes.csv) must live in the "
             "same folder.",
    )
    ap.add_argument(
        "--db", choices=DB_CHOICES, default="local",
        help="Target DB. local -> local Postgres sandbox via psycopg "
             "(SUPABASE_LOCAL_URI / SUPABASE_LOCAL_HOST+PORT). "
             "staging/prod -> Supabase REST (SUPABASE_URL/KEY).",
    )
    ap.add_argument(
        "--dotenv", type=str, default=None,
        help="Path to .env (defaults to ml-data-pipeline/.env if present). "
             "Connection URIs and supabase URL/key are read from here unless "
             "overridden by --url/--key.",
    )
    ap.add_argument(
        "--url", type=str, default=None,
        help="[staging/prod only] Supabase URL override.",
    )
    ap.add_argument(
        "--key", type=str, default=None,
        help="[staging/prod only] Supabase key override.",
    )
    ap.add_argument(
        "--cluster", choices=CLUSTER_CHOICES, default=None,
        help="Storage location where processed data resides; sets "
             "exists_<cluster>=true on each new prepared row.",
    )
    args = ap.parse_args()

    _, failed = add_synthetic_metadata_to_db(
        args.metadata_file,
        db=args.db,
        dotenv=args.dotenv,
        cluster=args.cluster,
        supabase_url=args.url,
        supabase_key=args.key,
    )
    if failed:
        sys.exit(1)

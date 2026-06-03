"""Reshape every synthetic zarr in a dataset so absolute upstream timepoints
align with zarr T-axis indices, writing the resharded dataset to a NEW
directory (the source tree is never modified).

Why this exists
---------------
The synthetic-data generator emits zarr arrays with T-axis densely packed:
``zarr.shape[0] == len(distinct_upstream_timepoints)``. But the bbox /
annotation CSVs label each frame with its absolute upstream timepoint
(e.g. ``{0,5,10,...,95}`` when the upstream pipeline sampled every 5th
frame of a 100-frame source). The downstream loader
(``cell_observatory_platform.data.datasets.pretrain_dataset_ray.LoaderActor._slice_hypercube``)
treats the ``time_start`` value from the DB as a direct zarr coordinate.

That means:
  - the loader does ``zarr[time_start:time_start+1, ...]`` -- if the bbox
    CSV labeled this frame "timepoint 5" but the frame sits at zarr index
    1, the loader silently fetches the wrong frame; and
  - the DB SQL ``time_size = LEAST(tile_n_timepoints - time_start, 1)``
    in ``aggregate_prepared_cube_channel_spatial_aggs_from_partitions``
    silently emits ``time_size = 0`` for every cube whose absolute
    ``time_start >= tile_n_timepoints``, which then gets filtered out by
    the annotation-agg refresh
    (``WHERE c.time_size = 1``). Result: ~76% of annotated cubes never
    reach the training query.

The fix is to make ``zarr[t, ...]`` mean exactly what the bbox CSV
labels "timepoint t" -- i.e., extend the T-axis to
``max(upstream_timepoints) + 1`` and place each existing frame at its
labeled position. Unwritten T positions stay as fill_value=0; in a
``sharding_indexed`` zarr3 array those positions don't materialize as
bytes on disk.

In addition to fixing the indexing, this reshape adopts the platform's
canonical TZYXC inner-chunk shape ``[1, *, *, *, C]`` (single timepoint
per inner read chunk), matching what
``cell_observatory_platform.data.io.create_zarr_spec`` would produce.
The existing synthetic arrays use ``inner_T = T`` which forces a full
T-volume read for any single-timepoint cube; the new layout lets the
loader fetch exactly one timepoint.

Empirically (one 1.81 GB tile, ``000x_005y_002z.zarr`` in iteration_2):
  - on-disk size: 1.81 GB -> 1.81 GB (1.00x).
  - wall time: ~65 s/tile single-threaded.

Safety
------
This script writes the resharded dataset to a NEW directory (``--dst-root``)
and never touches ``--src-root``. The destination tree mirrors the source
layout exactly:

    <src-root>/<DATE>/<fish>/<roi>/<tile>.zarr
    <dst-root>/<DATE>/<fish>/<roi>/<tile>.zarr

Sibling CSVs next to each tile zarr
(``<stem>_mask_bbox.csv``, ``<stem>_mem_occratio.csv``,
``<stem>_cytosol_occratio.csv``, ``<stem>_staging_annotations_3d.csv``,
``<stem>_cytosol_fp_expressions.csv``) are copied verbatim into the
destination -- the downstream ingest reads them from the same directory
as the zarr.

If a destination tile already exists the script refuses to clobber it
unless ``--overwrite`` is passed. ``--dry-run`` reports the planned
work without writing anything.

Per-tile mapping
----------------
For each tile we read its ``<stem>_mask_bbox.csv``, take the sorted
unique ``timepoint`` values, and require ``len(unique) ==
zarr.shape[0]``. Storage order in the source zarr T-axis is assumed to
be ascending in the upstream timepoint value (the convention emitted
by ``gen_membrane_zarr.create_zarr_spec``). The mapping is then
``zarr_idx i -> upstream_t = upstream_timepoints[i]`` and
``T_new = max(upstream_timepoints) + 1``.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import tensorstore as ts


# Sentinel file written into the destination tile dir AFTER both the zarr
# data and the sidecar CSVs are flushed. Its presence is the only "this
# tile is complete" signal we trust on resume. If the script is killed
# mid-write the sentinel is absent and the tile is automatically redone.
_DONE_SENTINEL = ".reshape_done"


logger = logging.getLogger(__name__)


# Sidecar CSV suffixes the ingest pipeline expects next to each tile zarr.
# Anything matching ``<stem>_*.csv`` next to the source zarr gets copied;
# we list the known names here only for documentation -- the copy step uses
# a glob so any future sidecar tags through automatically.
_SIDECAR_SUFFIXES = (
    "_mask_bbox.csv",
    "_mem_occratio.csv",
    "_cytosol_occratio.csv",
    "_staging_annotations_3d.csv",
    "_cytosol_fp_expressions.csv",
)


def discover_tile_zarrs(root: Path) -> list[Path]:
    """Walk ``<root>/<DATE>/<fish>/<roi>/<tile>.zarr``.

    The synthetic dataset is laid out as ``<DATE_dataset>/<fish>/<roi>/<tile>.zarr``
    with sidecar CSVs in the same ``<roi>`` directory.
    """
    out: list[Path] = []
    if not root.is_dir():
        raise FileNotFoundError(root)
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for fish_dir in sorted(date_dir.iterdir()):
            if not fish_dir.is_dir():
                continue
            for roi_dir in sorted(fish_dir.iterdir()):
                if not roi_dir.is_dir():
                    continue
                for tile_dir in sorted(roi_dir.iterdir()):
                    if tile_dir.is_dir() and tile_dir.suffix == ".zarr":
                        out.append(tile_dir)
    return out


def upstream_timepoints_for_tile(tile_zarr: Path) -> list[int]:
    """Sorted unique upstream timepoint values from the tile's bbox CSV.

    Storage order in the zarr T-axis is assumed to be ascending in the
    upstream timepoint value -- i.e. ``zarr[i, ...]`` is the i-th
    smallest distinct timepoint in the CSV. This matches what
    ``gen_membrane_zarr.create_zarr_spec`` produces.
    """
    stem = tile_zarr.with_suffix("").name
    bbox_csv = tile_zarr.parent / f"{stem}_mask_bbox.csv"
    if not bbox_csv.exists():
        raise FileNotFoundError(
            f"{tile_zarr}: expected sibling bbox CSV not found at {bbox_csv}"
        )
    df = pd.read_csv(bbox_csv, usecols=["timepoint"])
    return sorted(df["timepoint"].astype(int).unique().tolist())


def _build_dst_spec(
    *,
    path: Path,
    dtype: str,
    T_new: int,
    Z: int,
    Y: int,
    X: int,
    C: int,
    shard_spatial: tuple[int, int, int],
    inner_spatial: tuple[int, int, int],
) -> dict[str, Any]:
    """Construct the destination zarr3 spec in the platform-canonical TZYXC
    layout: a single T-shard covering the full T axis, inner read-chunk
    T=1 so single-timepoint cube reads don't pay for whole-T decode."""
    return {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": str(path)},
        "create": True,
        "delete_existing": True,
        "metadata": {
            "data_type": dtype,
            "shape": [T_new, Z, Y, X, C],
            "chunk_grid": {
                "name": "regular",
                "configuration": {
                    "chunk_shape": [T_new, Z, shard_spatial[1], shard_spatial[2], C],
                },
            },
            "codecs": [
                {
                    "name": "sharding_indexed",
                    "configuration": {
                        "chunk_shape": [1, Z, inner_spatial[1], inner_spatial[2], C],
                        "codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {
                                "name": "blosc",
                                "configuration": {
                                    "cname": "zstd",
                                    "clevel": 1,
                                    "blocksize": 0,
                                    "shuffle": "shuffle",
                                },
                            },
                        ],
                        "index_codecs": [
                            {"name": "bytes", "configuration": {"endian": "little"}},
                            {"name": "crc32c"},
                        ],
                        "index_location": "end",
                    },
                }
            ],
            "fill_value": 0,
        },
    }


def _copy_sidecar_csvs(src_tile_zarr: Path, dst_tile_zarr: Path) -> list[str]:
    """Copy every ``<stem>_*.csv`` next to the source zarr into the dst
    zarr's parent directory. Returns the list of copied filenames (relative
    names, for logging). Existing destination CSVs are overwritten.
    """
    stem = src_tile_zarr.with_suffix("").name
    src_dir = src_tile_zarr.parent
    dst_dir = dst_tile_zarr.parent
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for sidecar in sorted(src_dir.glob(f"{stem}_*.csv")):
        target = dst_dir / sidecar.name
        shutil.copyfile(sidecar, target)
        copied.append(sidecar.name)
    return copied


def _is_tile_complete(dst_tile_zarr: Path) -> bool:
    """A destination tile counts as complete only if both zarr.json exists
    AND we wrote the ``_DONE_SENTINEL`` file next to it. Anything else
    (zarr.json present but no sentinel, half-flushed shards, no sidecars)
    is treated as an interrupted run and will be redone on resume.
    """
    if not dst_tile_zarr.is_dir():
        return False
    if not (dst_tile_zarr / "zarr.json").is_file():
        return False
    return (dst_tile_zarr.parent / f"{dst_tile_zarr.stem}{_DONE_SENTINEL}").is_file()


def reshape_one_tile(
    src_tile_zarr_str: str,
    dst_tile_zarr_str: str,
    dry_run: bool = False,
    overwrite: bool = False,
    inner_concurrency: int = 0,
) -> dict[str, Any]:
    """Reshape a single tile zarr from src to dst. Returns a status dict.

    ``inner_concurrency``: max number of per-frame writes to keep in flight
    concurrently within this tile. ``0`` means "all frames at once" (which
    is fine for the synthetic dataset's T_old in [6, 20] -- each in-flight
    write only buffers one ZYXC frame inside tensorstore's executor).
    Strings used for path arguments to keep this picklable in case future
    callers want a ProcessPoolExecutor.
    """
    src_tile_zarr = Path(src_tile_zarr_str)
    dst_tile_zarr = Path(dst_tile_zarr_str)
    upstream_tps = upstream_timepoints_for_tile(src_tile_zarr)

    src = ts.open(
        {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": str(src_tile_zarr)},
        }
    ).result()
    src_shape = tuple(int(d) for d in src.shape)
    T_old, Z, Y, X, C = src_shape

    if len(upstream_tps) != T_old:
        raise ValueError(
            f"{src_tile_zarr}: bbox CSV has {len(upstream_tps)} distinct timepoints "
            f"but zarr T-axis is {T_old}; refusing to reshape -- the dense-rank "
            f"mapping requires CSV timepoints to enumerate exactly the zarr frames"
        )
    mapping = dict(enumerate(upstream_tps))  # zarr_idx -> upstream_t
    T_new = max(mapping.values()) + 1

    info: dict[str, Any] = {
        "src": str(src_tile_zarr),
        "dst": str(dst_tile_zarr),
        "src_shape": src_shape,
        "T_old": T_old,
        "T_new": T_new,
        "mapping_head": list(mapping.items())[:5],
        "mapping_tail": list(mapping.items())[-3:],
    }

    shard_spatial = tuple(int(d) for d in src.chunk_layout.write_chunk.shape[1:4])
    inner_spatial = tuple(int(d) for d in src.chunk_layout.read_chunk.shape[1:4])
    info["shard_spatial"] = shard_spatial
    info["inner_spatial"] = inner_spatial

    sentinel = dst_tile_zarr.parent / f"{dst_tile_zarr.stem}{_DONE_SENTINEL}"

    if _is_tile_complete(dst_tile_zarr) and not overwrite:
        info["status"] = "skipped (already complete; pass --overwrite to redo)"
        return info

    if dry_run:
        info["status"] = "dry-run"
        return info

    dst_tile_zarr.parent.mkdir(parents=True, exist_ok=True)
    # An existing dst dir without a sentinel is an interrupted prior run --
    # wipe it and start clean. With --overwrite we wipe regardless.
    if dst_tile_zarr.exists():
        shutil.rmtree(dst_tile_zarr)
    if sentinel.exists():
        sentinel.unlink()

    # NOTE: tensorstore.dtype's __str__ returns 'dtype("uint16")' which is
    # not a valid zarr3 ``data_type`` string. Use the ``.name`` attribute
    # ('uint16') instead.
    dst_spec = _build_dst_spec(
        path=dst_tile_zarr,
        dtype=src.dtype.name,
        T_new=T_new, Z=Z, Y=Y, X=X, C=C,
        shard_spatial=shard_spatial,
        inner_spatial=inner_spatial,
    )
    dst = ts.open(dst_spec).result()

    # Pipeline per-frame writes via tensorstore Futures. Each
    # dst[t:t+1].write(src[i:i+1]) returns a Future immediately; tensorstore's
    # C++ executor decodes the source shard and encodes the destination shard
    # in parallel for every in-flight frame. We only `.result()` at the end
    # (or in flush groups when inner_concurrency caps in-flight count) so the
    # whole tile becomes one wave of async I/O instead of T_old serial round
    # trips through Python.
    t0 = time.time()
    in_flight: list[ts.Future] = []
    limit = inner_concurrency if inner_concurrency > 0 else len(mapping)
    for i, t in mapping.items():
        fut = dst[t : t + 1, :, :, :, :].write(src[i : i + 1, :, :, :, :])
        in_flight.append(fut)
        if len(in_flight) >= limit:
            for f in in_flight:
                f.result()
            in_flight.clear()
    for f in in_flight:
        f.result()
    elapsed = time.time() - t0

    sidecars = _copy_sidecar_csvs(src_tile_zarr, dst_tile_zarr)
    # Sentinel goes down LAST, after sidecars are flushed -- so any kill
    # between the zarr write and this line leaves an "incomplete" tile that
    # a future invocation will redo automatically.
    sentinel.touch()

    def _disk_bytes(p: Path) -> int:
        return sum(
            os.path.getsize(os.path.join(d, f))
            for d, _, fs in os.walk(p)
            for f in fs
        )

    src_bytes = _disk_bytes(src_tile_zarr)
    dst_bytes = _disk_bytes(dst_tile_zarr)

    info.update({
        "status": "ok",
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "size_ratio": round(dst_bytes / src_bytes, 3) if src_bytes else None,
        "elapsed_s": round(elapsed, 1),
        "sidecars_copied": sidecars,
    })
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--src-root",
        required=True,
        help=(
            "Source synthetic-data root containing "
            "<DATE>/<fish>/<roi>/<tile>.zarr trees, e.g. "
            "/clusterfs/vast/forsynthetic/benchmark_tests/iteration2_martin/synthetic_data_iteration_2"
        ),
    )
    ap.add_argument(
        "--dst-root",
        required=True,
        help=(
            "Destination root for the resharded dataset. The directory tree "
            "from --src-root is mirrored beneath this path; "
            "sidecar <stem>_*.csv files are copied verbatim. The source tree "
            "is never modified."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "number of tiles to reshape concurrently (default: 8). Threads, "
            "not processes -- tensorstore releases the GIL during read/write "
            "and uses its own C++ executor, so threads are strictly cheaper "
            "than processes here (no fork/import overhead, no pickling)."
        ),
    )
    ap.add_argument(
        "--inner-concurrency",
        type=int,
        default=0,
        help=(
            "max in-flight per-frame writes inside one tile (0 = all frames "
            "at once). Each in-flight write only buffers a single ZYXC frame "
            "in tensorstore's executor, so for T_old <= 20 the default is "
            "safe. Set to e.g. 4 if memory is tight."
        ),
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="only process the first N tiles (0 = all); useful for staged rollout",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "validate every tile's bbox CSV against its zarr and print the "
            "(zarr_idx -> upstream_t) mapping and planned destination path "
            "without writing anything"
        ),
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "overwrite destination tile zarrs that already exist "
            "(default: skip with a warning)"
        ),
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    src_root = Path(args.src_root).resolve()
    dst_root = Path(args.dst_root).resolve()
    if dst_root == src_root:
        raise SystemExit("--dst-root must not equal --src-root (this script never writes in place)")
    if src_root in dst_root.parents:
        raise SystemExit(
            f"--dst-root ({dst_root}) is inside --src-root ({src_root}); "
            f"choose a destination outside the source tree"
        )

    tiles = discover_tile_zarrs(src_root)
    if args.limit:
        tiles = tiles[: args.limit]
    logger.info("found %d tile zarrs under %s", len(tiles), src_root)
    logger.info("destination root: %s", dst_root)
    if not tiles:
        logger.warning("nothing to do")
        return 0

    # Plan: for each src tile compute the mirrored destination path.
    plan: list[tuple[Path, Path]] = []
    for src_tile in tiles:
        rel = src_tile.relative_to(src_root)
        dst_tile = dst_root / rel
        plan.append((src_tile, dst_tile))

    failures: list[tuple[Path, str]] = []
    successes = 0
    skipped = 0

    def _run_one(src_tile: Path, dst_tile: Path) -> dict[str, Any]:
        return reshape_one_tile(
            str(src_tile), str(dst_tile),
            dry_run=args.dry_run, overwrite=args.overwrite,
            inner_concurrency=args.inner_concurrency,
        )

    if args.workers <= 1 or args.dry_run:
        for src_tile, dst_tile in plan:
            try:
                info = _run_one(src_tile, dst_tile)
                logger.info("%s", info)
                if str(info.get("status", "")).startswith("skipped"):
                    skipped += 1
                else:
                    successes += 1
            except Exception as e:
                logger.exception("failed: %s", src_tile)
                failures.append((src_tile, repr(e)))
    else:
        # Threads, not processes. tensorstore's read/write release the GIL
        # and dispatch to a C++ executor, so the Python "worker" only owns
        # a Future handle. Threads avoid the ~1s fork+import cost per
        # ProcessPoolExecutor child and let `kbd interrupt` propagate
        # cleanly (a SIGINT raises KeyboardInterrupt in the main thread
        # which `ThreadPoolExecutor.__exit__` cooperates with).
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_run_one, s, d): s for s, d in plan}
            for fut in as_completed(futs):
                src_tile = futs[fut]
                try:
                    info = fut.result()
                    logger.info("%s", info)
                    if str(info.get("status", "")).startswith("skipped"):
                        skipped += 1
                    else:
                        successes += 1
                except Exception as e:
                    logger.exception("failed: %s", src_tile)
                    failures.append((src_tile, repr(e)))

    logger.info(
        "summary: %d ok, %d skipped, %d failed (of %d tiles)",
        successes, skipped, len(failures), len(tiles),
    )
    if failures:
        for t, e in failures:
            logger.error("  FAIL %s -> %s", t, e)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

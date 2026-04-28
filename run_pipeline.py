from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import re
import sys
import uuid
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("run_pipeline")


ROI_PATH_PREFIXES = [
    "/clusterfs/abc2/CellObservatory",
    "/clusterfs/vast/abcabc"
]


def _normalize_roi_path(roi_path: str) -> str:
    """Ensure roi_path is an absolute Linux path under a known prefix.

    Tries each prefix in ROI_PATH_PREFIXES and returns the first that
    exists on disk.  Falls back to the first prefix if none exist (e.g.
    during dry-run on a node without the data mounted).
    """
    path = str(roi_path).strip().replace("\\", "/")

    for prefix in ROI_PATH_PREFIXES:
        if path.startswith(prefix):
            return path

    path = re.sub(r"^[A-Za-z]:/+", "", path)
    path = re.sub(r"^abcabc/+", "", path, flags=re.IGNORECASE)
    path = path.lstrip("/")

    for prefix in ROI_PATH_PREFIXES:
        candidate = f"{prefix}/{path}"
        if os.path.isdir(candidate) and _has_tiff_files(candidate):
            return candidate

    return f"{ROI_PATH_PREFIXES[0]}/{path}"


def _has_tiff_files(directory: str) -> bool:
    """Return True as soon as any .tif/.tiff is found (no full tree walk)."""
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".tif", ".tiff")):
                return True
    return False


def _normalize_single_token_channels(
    roi_id: str,
    biological_channel_to_tiff_tokens: dict,
    pipeline_run_id: str,
) -> tuple[dict[str, str] | None, dict | None]:
    normalized: dict[str, str] = {}

    for bio_name, value in (biological_channel_to_tiff_tokens or {}).items():
        if isinstance(value, list):
            tokens = [str(v) for v in value if v]
            if len(tokens) != 1:
                message = (
                    f"Skipped ROI before scan: biological channel '{bio_name}' has "
                    f"{len(tokens)} tiff tokens; expected exactly 1"
                )
                logger.info("Skipping ROI %s: channel %s has %d tiff tokens", roi_id, bio_name, len(tokens))
                return None, {
                    "roi_acquisition_id": roi_id,
                    "tile_name": "000x_000y_000z",
                    "status": "failed",
                    "error_messages": [message],
                    "scan_metadata": {
                        "skip_reason": "multiple_tiff_tokens_per_biological_channel",
                        "biological_channel": bio_name,
                        "n_tiff_tokens": len(tokens),
                        "tiff_tokens": tokens,
                    },
                    "pipeline_run_id": pipeline_run_id,
                }
            normalized[str(bio_name)] = tokens[0]
        elif value:
            normalized[str(bio_name)] = str(value)

    return normalized, None


_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _load_roi_ids_from_file(path: str) -> list[str]:
    """Load ROI acquisition IDs from a plain-text or CSV file.

    Accepts two formats:
    * One UUID per line (plain text).
    * A CSV with a ``raw_roi_acquisition_id`` header column.

    Raises ValueError if the file is neither.
    """
    text = Path(path).read_text().strip()
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"--roi-ids-file {path} is empty")

    header = lines[0]

    # CSV with raw_roi_acquisition_id column
    if "," in header:
        cols = [c.strip().lower() for c in header.split(",")]
        if "raw_roi_acquisition_id" not in cols:
            raise ValueError(
                f"--roi-ids-file {path} looks like a CSV but has no "
                f"'raw_roi_acquisition_id' column. Header: {header}"
            )
        idx = cols.index("raw_roi_acquisition_id")
        ids = []
        for line in lines[1:]:
            parts = line.split(",")
            if len(parts) > idx:
                candidate = parts[idx].strip().strip("'\"")
                if _UUID_RE.match(candidate):
                    ids.append(candidate)
        if not ids:
            raise ValueError(
                f"--roi-ids-file {path} has a raw_roi_acquisition_id column "
                f"but no valid UUIDs were found"
            )
        return ids

    # Plain text: one UUID per line
    ids = [line.strip() for line in lines if _UUID_RE.match(line.strip())]
    if not ids:
        raise ValueError(
            f"--roi-ids-file {path} contains no valid UUIDs. "
            f"Expected one UUID per line or a CSV with raw_roi_acquisition_id column."
        )
    return ids


def _load_prepared_ids_from_file(path: str) -> list[int]:
    """Load prepared IDs from one-int-per-line text or CSV.

    CSV files may use either a ``prepared_id`` or ``id`` column. Plain text
    files may include comments starting with ``#``.
    """
    lines = Path(path).read_text().splitlines()
    rows = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]
    if not rows:
        raise ValueError(f"--prepared-ids-file {path} is empty")

    first_cols = [c.strip().lower() for c in rows[0].split(",")]
    if "prepared_id" in first_cols or "id" in first_cols:
        idx = first_cols.index("prepared_id") if "prepared_id" in first_cols else first_cols.index("id")
        ids: list[int] = []
        for row in rows[1:]:
            parts = row.split(",")
            if len(parts) > idx and parts[idx].strip():
                ids.append(int(parts[idx].strip()))
        if not ids:
            raise ValueError(f"--prepared-ids-file {path} has no prepared IDs")
        return ids

    ids = []
    for row in rows:
        first_field = row.split(",", 1)[0].strip()
        ids.append(int(first_field))
    return ids


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="DB-driven ML data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    db_group = ap.add_argument_group("Database")
    db_group.add_argument("--db", choices=["local", "staging", "prod"], default="prod")
    db_group.add_argument(
        "--dotenv",
        type=str,
        default=None,
        help="Path to .env file (defaults to repo-root .env if present)",
    )

    store_group = ap.add_argument_group("Persistence")
    store_group.add_argument("--store", choices=["json", "db"], default="db")
    store_group.add_argument("--store-dir", type=str, default=None, help="Base dir for json store")

    ch_group = ap.add_argument_group("Channel mapping")
    ch_group.add_argument(
        "--channels", type=lambda s: [c.strip() for c in s.split(",")],
        required=True,
        help="Biological channels in priority order, e.g. 'membrane,histone'. "
             "Alias+substring matching against DB bio channel names.",
    )

    proc_group = ap.add_argument_group("Processing")
    proc_group.add_argument("--output-folder", type=str, required=True)
    proc_group.add_argument("--log-dir", type=str, required=True)
    proc_group.add_argument(
        "--cube-shape", type=lambda s: tuple(map(int, s.split(","))),
        default=(128, 128, 128),
    )
    proc_group.add_argument("--batch-size", type=int, default=16)
    proc_group.add_argument("--output-zarr-version", type=str, default="zarr3")
    proc_group.add_argument("--background-folder", type=str, default="")
    proc_group.add_argument("--add-support-ratio-metadata", action="store_true")
    proc_group.add_argument(
        "--cluster", type=str, required=True,
        default="abc",
        choices=["prfs", "aws", "oak", "abc", "nersc", "gcp"],
        help="Storage location where processed data resides.",
    )

    sel_group = ap.add_argument_group("Selection")
    sel_group.add_argument("--dry-run", action="store_true")
    sel_group.add_argument("--all", action="store_true", dest="select_all")
    sel_group.add_argument("--roi-ids", type=lambda s: s.split(","), default=None)
    sel_group.add_argument(
        "--roi-ids-file", type=str, default=None,
        help="Path to a file with ROI IDs: one UUID per line, or a CSV "
             "with a 'raw_roi_acquisition_id' column.",
    )
    sel_group.add_argument("--proportion", type=float, default=None)
    sel_group.add_argument("--max-rois", type=int, default=None)
    sel_group.add_argument("--max-total-size-gb", type=float, default=None)

    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument(
        "--skip-recently-failed", action="store_true",
        help="Skip ROIs with recent scan failures"
    )
    ap.add_argument("--failed-lookback-days", type=int, default=7)

    meta_group = ap.add_argument_group("Metadata-only ingest")
    meta_group.add_argument(
        "--skip-processing", action="store_true",
        help="Skip all heavy processing jobs (CSC/decon/DSR/conversion/zarr). "
             "Rebuild metadata from raw scan + existing metadata.json on disk.",
    )
    meta_group.add_argument(
        "--output-date-ymd", type=str, default=None,
        help="Comma-separated YYYY,MM,DD locating the already-processed output folder. "
             "Required when --skip-processing is set.",
    )

    refresh_group = ap.add_argument_group("Cache refresh")
    refresh_group.add_argument(
        "--refresh-statement-timeout",
        type=str,
        default="2h",
        help="Postgres statement_timeout for each per-ROI cache refresh "
             "(e.g. '20min', '2h', or '0' to disable). Default: 2h.",
    )
    refresh_group.add_argument(
        "--refresh-fail-fast",
        action="store_true",
        help="Stop immediately if one prepared_id cache refresh fails. "
             "Default behavior rolls back the failed prepared_id, continues, "
             "and writes failed IDs to log_dir.",
    )

    return ap.parse_args(argv)


def parse_refresh_cache_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Refresh prepared cache artifacts for existing prepared IDs",
    )
    ap.add_argument("--db", choices=["local", "staging", "prod"], default="prod")
    ap.add_argument(
        "--dotenv",
        type=str,
        default=None,
        help="Path to .env file (defaults to repo-root .env if present)",
    )
    ap.add_argument(
        "--prepared-ids",
        type=lambda s: [int(x.strip()) for x in s.split(",") if x.strip()],
        default=None,
        help="Comma-separated prepared IDs, e.g. 23,24,25",
    )
    ap.add_argument(
        "--prepared-ids-file",
        type=str,
        default=None,
        help="File with one prepared_id per line, or CSV with prepared_id/id column.",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for refresh failure logs.",
    )
    ap.add_argument(
        "--refresh-statement-timeout",
        type=str,
        default="2h",
        help="Postgres statement_timeout per prepared_id refresh. Use 0 to disable.",
    )
    ap.add_argument(
        "--refresh-fail-fast",
        action="store_true",
        help="Stop on first refresh failure. Default is continue and log failed IDs.",
    )
    return ap.parse_args(argv)


def refresh_cache_main(argv=None) -> int:
    args = parse_refresh_cache_args(argv)

    from pipeline.db_client import PipelineDBClient
    from pipeline.store import create_store

    prepared_ids: list[int] = []
    if args.prepared_ids:
        prepared_ids.extend(args.prepared_ids)
    if args.prepared_ids_file:
        prepared_ids.extend(_load_prepared_ids_from_file(args.prepared_ids_file))

    prepared_ids = sorted(set(prepared_ids))
    if not prepared_ids:
        raise ValueError("Provide --prepared-ids or --prepared-ids-file")

    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Refreshing cache artifacts for %d prepared IDs: %s",
        len(prepared_ids),
        prepared_ids,
    )

    db_client = PipelineDBClient(mode=args.db, dotenv_path=args.dotenv)
    store = create_store("db", db_client=db_client)
    failed_ids = store.refresh_cache_artifacts(
        prepared_ids,
        statement_timeout=args.refresh_statement_timeout,
        continue_on_error=not args.refresh_fail_fast,
    )

    if failed_ids:
        failure_path = Path(args.log_dir) / "refresh_cache_failures.txt"
        failure_path.write_text("\n".join(str(pid) for pid in failed_ids) + "\n")
        logger.error(
            "Cache refresh failed for %d prepared IDs: %s. Wrote %s",
            len(failed_ids),
            failed_ids,
            failure_path,
        )
        return 2

    logger.info("Cache refresh completed successfully for all %d prepared IDs", len(prepared_ids))
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)

    from pipeline.db_client import PipelineDBClient
    from pipeline.store import create_store
    from pipeline.discovery import discover_unprocessed_rois
    from pipeline.channel_mapping import (
        resolve_channel_mapping,
        resolve_tiff_tokens_for_mapping,
    )
    from pipeline.scanner import scan_roi, scan_result_to_log_entries
    from pipeline.estimator import estimate_roi_size, format_dry_run_summary
    from pipeline.selector import select_rois_cli
    from pipeline.planner import build_processing_plan
    from pipeline.executor import execute_plan
    from pipeline.preprocessing import load_profiles, match_profile

    run_id = args.run_id or str(uuid.uuid4())[:8]
    logger.info("Pipeline run_id=%s", run_id)

    preprocessing_profiles, channel_aliases = load_profiles()

    db_client = PipelineDBClient(mode=args.db, dotenv_path=args.dotenv)

    store_kwargs = {"db_client": db_client} if args.store == "db" else {"store_dir": args.store_dir}
    store = create_store(args.store, **store_kwargs)

    # --- Discovery ---
    rois_table = discover_unprocessed_rois(
        db_client,
    )
    if rois_table.num_rows == 0:
        logger.info("No unprocessed ROIs found. Done.")
        return 0

    # --- ROI ID pre-filter (before scanning) ---
    if args.roi_ids_file:
        file_ids = _load_roi_ids_from_file(args.roi_ids_file)
        args.roi_ids = (args.roi_ids or []) + file_ids
        logger.info("Loaded %d ROI IDs from %s", len(file_ids), args.roi_ids_file)

    if args.roi_ids:
        roi_id_set = set(args.roi_ids)
        mask = [
            str(rois_table.column("roi_acquisition_id")[i].as_py()) in roi_id_set
            for i in range(rois_table.num_rows)
        ]
        rois_table = rois_table.filter(mask)
        logger.info(
            "Pre-filtered to %d ROIs matching --roi-ids/--roi-ids-file "
            "(%d IDs requested)", rois_table.num_rows, len(roi_id_set),
        )
        if rois_table.num_rows == 0:
            logger.info("No ROIs match the requested IDs. Done.")
            return 0

    # --- Scan, resolve mappings, estimate ---
    scan_results = []
    channel_mappings = []
    tiff_token_mappings = []
    size_estimates = []
    preprocessing_configs = []
    all_log_entries = []

    for i in tqdm(range(rois_table.num_rows), total=rois_table.num_rows, desc="Scanning ROIs"):
        roi_id = str(rois_table.column("roi_acquisition_id")[i].as_py())
        # HACK: some problems with paths in carp db
        roi_path = _normalize_roi_path(str(rois_table.column("roi_path")[i].as_py()))
        bio_tokens_raw = rois_table.column("biological_channel_to_tiff_tokens")[i].as_py()
        if isinstance(bio_tokens_raw, str):
            bio_tokens = json.loads(bio_tokens_raw)
        else:
            bio_tokens = bio_tokens_raw or {}
        bio_tokens, skip_log_entry = _normalize_single_token_channels(roi_id, bio_tokens, run_id)
        if skip_log_entry is not None:
            all_log_entries.append(skip_log_entry)
            continue

        ch_mapping = resolve_channel_mapping(args.channels, bio_tokens, channel_aliases)
        if ch_mapping is None:
            logger.warning("Skipping ROI %s: cannot satisfy channel mapping", roi_id)
            continue

        profile = match_profile(
            args.channels, roi_path, preprocessing_profiles, channel_aliases,
        )
        if profile is None:
            logger.warning(
                "Skipping ROI %s: no preprocessing profile for channels %s",
                roi_id, sorted(ch_mapping.values()),
            )
            continue
        preproc = profile.to_dataset_fields()
        logger.info("ROI %s matched preprocessing profile %r", roi_id, profile.name)

        scan = scan_roi(roi_id, roi_path, bio_tokens, ch_mapping)
        log_entries = scan_result_to_log_entries(scan, pipeline_run_id=run_id)
        all_log_entries.extend(log_entries)

        tok_mapping = resolve_tiff_tokens_for_mapping(ch_mapping, bio_tokens)
        estimate = estimate_roi_size(
            scan,
            n_channels=len(ch_mapping),
        )

        scan_results.append(scan)
        channel_mappings.append(ch_mapping)
        tiff_token_mappings.append(tok_mapping)
        size_estimates.append(estimate)
        preprocessing_configs.append(preproc)

    # Persist scan logs
    if all_log_entries:
        store.write_scan_log(all_log_entries)

    if not scan_results:
        logger.info("No ROIs passed scanning. Done.")
        return 0

    # --- Dry-run summary ---
    summary = format_dry_run_summary(size_estimates, output_folder=args.output_folder)
    print(summary)

    if args.dry_run:
        logger.info("Dry-run mode. Exiting.")
        return 0

    # --- Selection ---
    selected = select_rois_cli(
        size_estimates,
        roi_ids=args.roi_ids,
        proportion=args.proportion,
        max_rois=args.max_rois,
        max_total_size_gb=args.max_total_size_gb,
        select_all=args.select_all,
    )
    if not selected:
        logger.info("No ROIs selected. Done.")
        return 0

    # Filter scan_results/mappings to selected
    selected_ids = {e.roi_acquisition_id for e in selected}
    sel_scans = []
    sel_mappings = []
    sel_tok_mappings = []
    sel_estimates = []
    sel_preprocs = []
    for scan, ch_map, tok_map, est, preproc in zip(
        scan_results, channel_mappings, tiff_token_mappings, size_estimates, preprocessing_configs,
    ):
        if scan.roi_acquisition_id in selected_ids:
            sel_scans.append(scan)
            sel_mappings.append(ch_map)
            sel_tok_mappings.append(tok_map)
            sel_estimates.append(est)
            sel_preprocs.append(preproc)

    # --- Plan ---
    plan = build_processing_plan(
        sel_scans,
        sel_estimates,
        sel_mappings,
        sel_tok_mappings,
        output_folder=args.output_folder,
        cube_shape=args.cube_shape,
        batch_size=args.batch_size,
        output_zarr_version=args.output_zarr_version,
        run_id=run_id,
        preprocessing_configs=sel_preprocs,
    )

    # Save plan to disk
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    plan_path = f"{args.log_dir}/plan_{run_id}.json"
    with open(plan_path, "w") as f:
        f.write(plan.to_json())
    logger.info("Plan saved to %s", plan_path)

    # --- Execute ---
    output_date_ymd = args.output_date_ymd.split(",") if args.output_date_ymd else None
    if args.skip_processing and output_date_ymd is None:
        logger.error("--skip-processing requires --output-date-ymd YYYY,MM,DD")
        return 1

    prepared_ids = execute_plan(
        plan,
        sel_scans,
        store,
        log_dir=args.log_dir,
        background_folder=args.background_folder,
        add_support_ratio_metadata=args.add_support_ratio_metadata,
        skip_processing=args.skip_processing,
        output_date_ymd=output_date_ymd,
        cluster=args.cluster,
        refresh_statement_timeout=args.refresh_statement_timeout,
        refresh_continue_on_error=not args.refresh_fail_fast,
    )

    logger.info("Pipeline complete. prepared_ids=%s", prepared_ids)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "refresh-cache":
        sys.exit(refresh_cache_main(sys.argv[2:]))
    sys.exit(main())

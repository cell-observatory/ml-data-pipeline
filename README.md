# ml-data-pipeline

DB-driven pipeline for discovering raw ROI datasets, validating TIFF layout, estimating output size, selecting ROIs, generating a processing plan, and (optionally) ingesting results into the acquisition DB.

## What it does

`run_pipeline.py` is the main entrypoint. It:

1. Connects to the acquisition DB (`local`, `staging`, or `prod`) via `pipeline.db_client.PipelineDBClient`.
2. Queries `raw_rois` for entries where `is_prepared = false`.
3. Optionally pre-filters to a specific set of ROIs (`--roi-ids` or `--roi-ids-file`).
4. Resolves the output channel mapping from `--channels` using alias + substring matching against each ROI's `biological_channel_to_tiff_tokens`.
5. Matches each ROI to a preprocessing profile from `preprocessing_profiles.json`.
6. Scans TIFF files in each ROI folder and validates tiles/timepoints.
7. Prints a dry-run size summary and prompts the user to select ROIs (`[a]ll` / `[s]elect by index` / `[p]roportion` / `[q]uit`).
8. Writes a JSON processing plan to `--log-dir/plan_<run_id>.json`.
9. For non-dry runs, executes the plan and ingests results into the configured store.

Two execution paths on the real run:

- **Full processing** (default): submits CSC unmixing / decon / DSR / training-image / Zarr-conversion jobs, then ingests their metadata into the DB.
- ***Metadata-only** (*`--skip-processing`*): skips every heavy job and instead reads existing* `metadata.json` *files under* `<output-folder>/YYYY/M/D/...` *to rebuild the DB rows. Intended for re-ingesting already-processed ROIs, e.g. after a schema change.*

## CLI flags

### Required

- `--channels membrane,histone` — biological channels in priority order. Alias + substring matching against DB bio channel names via `preprocessing_profiles.json` → `channel_aliases`.
- `--output-folder PATH` — root for processed outputs (e.g. `/clusterfs/vast/Data/cell_observatory_training_datasets`).
- `--log-dir PATH` — where `plan_<run_id>.json` (and per-ROI logs) are written.
- `--cluster {prfs,aws,oak,abc,nersc,gcp}` — storage location of the processed data; sets the matching `exists_<cluster>` flag on each new `prepared` row.

### Database & persistence

- `--db {local,staging,prod}` (default `prod`) — which acquisition DB to connect to.
- `--dotenv PATH` — path to a `.env` file. Defaults to the repo-root `.env` if present.
- `--store {json,db}` (default `db`) — write target.
  - `json`: writes scan/prepared artifacts under `--store-dir` (safe for inspection).
  - `db`: writes directly to the DB backend implied by `--db` (local Postgres sandbox for `local`, Supabase REST for `staging` / `prod`).
- `--store-dir PATH` — required when `--store json`.

### Selection

- `--dry-run` — exit after the size summary, before ingestion.
- `--roi-ids id1,id2,...` — comma-separated ROI UUIDs.
- `--roi-ids-file PATH` — one UUID per line, or a CSV with a `raw_roi_acquisition_id` column. Combined with `--roi-ids` if both are given.
- `--all` — auto-answer the selector prompt with "all".
- `--proportion FLOAT` — keep a random fraction.
- `--max-rois INT`, `--max-total-size-gb FLOAT` — cap the selection size.

### Processing

- `--cube-shape 128,128,128` (default)
- `--batch-size 16` (default)
- `--output-zarr-version zarr3` (default)
- `--background-folder PATH` — optional, passed through to processing jobs.
- `--add-support-ratio-metadata` — compute the extra support-ratio metric during processing.

### Metadata-only ingest

- `--skip-processing` — skip all heavy jobs; rebuild metadata from raw scan + existing `metadata.json` on disk.
- `--output-date-ymd YYYY,MM,DD` — required with `--skip-processing`; locates the already-processed date folder under `--output-folder`.

### Misc

- `--run-id STRING` — override the auto-generated run id.
- `--skip-recently-failed`, `--failed-lookback-days INT` — filter out ROIs with recent scan failures.

## Environment

- Python packages: see `requirements.txt` (`connectorx`, `cpptiff`, `numpy`, `orjson`, `pandas`, `psycopg[binary]`, `PyPetaKit5D`, `pyarrow`, `python-dotenv`, `scikit-image`, `scipy`, `supabase`, `tensorstore`, `tqdm`, `zarr`).
- For full (non-`--skip-processing`) cluster execution: MATLAB, Slurm, `PyPetaKit5D`, `tensorstore`, `zarr`. `PyPetaKit5D` is lazy-imported, so `--skip-processing` works without it installed.
- A repo-root `.env` is loaded automatically by the DB client; override with `--dotenv`. An example is at `.env.example`.

Environment variables used by the DB client:

- `SUPABASE_LOCAL_URI` (or `SUPABASE_LOCAL_HOST` + `SUPABASE_LOCAL_PORT`)
- `SUPABASE_STAGING_URI`, `SUPABASE_PROD_URI`
- `SUPABASE_URL`, `SUPABASE_KEY` (used by the Supabase REST client for writes against staging / prod)

### Local sandbox startup

`--db local` assumes a local Postgres sandbox is already running with the full `acquistion_db` migrations applied (including `refresh_prepared_cache_artifacts(...)` and the `prepared_cube_`* aggregate tables).

Two startup flows live in the platform repo:

- `cell_observatory_platform/scripts/db/local_start_sandbox.sh` — the standard platform flow (start a Supabase-managed sandbox from the `acquistion_db` repo, export `SUPABASE_LOCAL_HOST` / `SUPABASE_LOCAL_URI`).

Pair one of those with the `ml-data-pipeline` local-mode commands below.

## Common commands

### Dry run against the local sandbox

```bash
python run_pipeline.py \
  --db local \
  --channels membrane,histone \
  --cluster abc \
  --store json \
  --store-dir /scratch/$USER/ml-data-pipeline/artifacts/local \
  --output-folder /scratch/$USER/ml-data-pipeline/output/local \
  --log-dir /scratch/$USER/ml-data-pipeline/logs \
  --dry-run --all
```

### Dry run against staging or prod metadata

Reads Supabase; writes only local JSON artifacts.

```bash
python run_pipeline.py \
  --db staging \
  --channels membrane,histone \
  --cluster abc \
  --store json \
  --store-dir /scratch/$USER/ml-data-pipeline/artifacts/staging \
  --output-folder /scratch/$USER/ml-data-pipeline/output/staging \
  --log-dir /scratch/$USER/ml-data-pipeline/logs \
  --dry-run --all
```

### Local DB write test, narrow ROI selection

```bash
python run_pipeline.py \
  --db local \
  --channels membrane,histone \
  --cluster abc \
  --store db \
  --output-folder /clusterfs/vast/Data/cell_observatory_training_datasets \
  --log-dir /tmp/$USER/ml-data-pipeline/logs \
  --roi-ids ROI_ID_1
```

### Metadata-only re-ingest (skip-processing)

Re-builds DB rows for already-processed ROIs by reading `metadata.json` under `--output-folder/YYYY/M/D/...`. Nothing is written to the output folder — this mode is read-only on disk.

```bash
python run_pipeline.py \
  --db local \
  --channels membrane,histone \
  --cluster abc \
  --skip-processing \
  --output-date-ymd 2025,10,1 \
  --roi-ids-file /clusterfs/nvme/hph/git_managed/databases/matched_roi_ids_2025_10_1.txt \
  --output-folder /clusterfs/vast/Data/cell_observatory_training_datasets \
  --log-dir /tmp/$USER/ml-data-pipeline/logs
```

Flip `--db local` → `--db staging` or `--db prod` to run the same ingest against Supabase. Because each date batch lives under a single `YYYY/M/D` prefix, run one invocation per date batch.

### Full processing run (Slurm / PetaKit5D required)

```bash
python run_pipeline.py \
  --db staging \
  --channels membrane,histone \
  --cluster abc \
  --output-folder /clusterfs/vast/Data/cell_observatory_training_datasets \
  --log-dir /tmp/$USER/ml-data-pipeline/logs \
  --all
```

## Outputs

- Dry-run summary printed to stdout.
- Per-ROI scan log rows in `roi_tile_scan_log` (DB) or JSON under `--store-dir/scan_logs` (JSON store).
- Plan file at `--log-dir/plan_<run_id>.json`.
- Non-dry, non-skip runs write `metadata.json` under `--output-folder/YYYY/M/D/<exp>/<fish>/<roi>/` plus the processed Zarr images.
- Non-dry runs ingest into `prepared`, `prepared_tiles`, and `prepared_cubes`, then call `refresh_prepared_cache_artifacts(prepared_id)` to repopulate the aggregate / view tables (`prepared_tiles_view_table`, `prepared_tile_channel_agg_*`, `prepared_tile_annotation_agg_1`, `prepared_cube_channel_agg_*`, `prepared_cube_annotation_agg_1_*`).

## Performance notes

- On `--db local`, `LocalPostgresStore.refresh_cache_artifacts` runs `ANALYZE` on `prepared`, `prepared_tiles`, `prepared_tiles_view_table`, and `prepared_cubes` before the per-ROI refresh loop. Without it, planner stats are stale right after a bulk ingest and the cube-level aggregators run roughly 2× slower.
- Cube inserts are batched (`CUBE_INSERT_BATCH = 10_000`) on both local and Supabase stores. On Supabase each batch is a round-trip over HTTPS, so ingest is significantly slower than local.
- The aggregate refresh is batched at the Python level: it runs once after the full ingest loop completes, not per ROI.

## Caveats

- Non-skip runs submit Slurm jobs and invoke MATLAB / PetaKit5D. Test with a narrow ROI selection before scaling up.
- Local DB mode assumes the sandbox has all `acquistion_db` migrations applied — in particular `refresh_prepared_cache_artifacts(...)` and the six `prepared_cube_channel_agg_*` tables.
- Staging / prod DB writes use the Supabase REST client; a service-role `SUPABASE_KEY` is required.
- `raw_rois.is_prepared` is set to `true` once ingestion succeeds, so re-running with the same `--roi-ids` will report zero matches. To re-ingest, reset the flag (and delete the corresponding `prepared` rows) in the DB first.

## Related scripts

- `submit_jobs.py` — Slurm / MATLAB / PetaKit5D job submission + metadata collection.
- `convert_files.py` — Zarr / training image conversion.
- `csc_unmixing.py` — chromatic shift correction and unmixing.
- `decon_dsr.py` — deconvolution, deskew, and rotate.
- `preprocessing_profiles.json` — per-dataset preprocessing configs and `channel_aliases` used by `--channels` matching.


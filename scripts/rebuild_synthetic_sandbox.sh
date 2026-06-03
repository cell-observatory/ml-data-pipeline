#!/usr/bin/env bash
# End-to-end rebuild of the local Postgres sandbox with the synthetic
# benchmark dataset.
#
# Steps:
#   1. (optional) regenerate channel_mapping.json from a *_cytosol_fp_expressions.csv
#   2. (optional) regenerate prepared.csv / prepared_tiles.csv / prepared_cubes.csv
#      from the synthetic data tree
#   3. (optional) wipe the existing scratch sandbox so the snapshot tarball is re-extracted
#   4. start the sandbox (start_local_sandbox.sh, pre-cleans stale locks)
#   5. (optional) purge any existing is_synthetic rows + their cache-aggregate rows
#   6. ingest the new prepared/tiles/cubes via add_synthetic_metadata_to_db.py
#   7. ANALYZE + CHECKPOINT for snapshot durability
#   8. capture verification SQL output
#   9. graceful shutdown (pg_ctl stop inside container, then apptainer instance stop)
#  10. pack a dated tarball
#  11. (optional) publish the tarball into ${DATABASE_DIR}
#
# All long-running steps print to stdout as they go. On any failure the
# script exits non-zero and leaves the sandbox running so you can poke
# at it; re-run with --skip-* flags to resume from the failing step.
#
# Usage example (full run, regenerating everything):
#
#   ./scripts/rebuild_synthetic_sandbox.sh \
#       --synth-root           /clusterfs/vast/forsynthetic/benchmark_tests/iteration2_martin/synthetic_data_iteration_2 \
#       --membrane-fp-name     tdmstaygold \
#       --fp-expressions-csv   /path/to/<some_tile>_cytosol_fp_expressions.csv \
#       --cluster              abc \
#       --publish-dir          /clusterfs/nvme/martinalvarez/databases
#
# Re-run after editing the channel_mapping JSON (skip the regen step):
#
#   ./scripts/rebuild_synthetic_sandbox.sh \
#       --synth-root         <root> \
#       --channel-mapping    /tmp/$USER/synth_csvs/channel_mapping.json \
#       --skip-mapping-regen \
#       --skip-fresh-sandbox \
#       --skip-publish

set -euo pipefail

# ------------------------------------------------------------------ paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_DIR="${ML_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
REPO_DIR="${REPO_DIR:-/clusterfs/nvme/martinalvarez/GitHub/cell_observatory_platform}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

# ------------------------------------------------------------------ defaults
SYNTH_ROOT=""
SERVER_FOLDER=""
MEMBRANE_FP_NAME=""
FP_EXPRESSIONS_CSV=""
CHANNEL_MAPPING=""
CSV_OUT_DIR=""
CLUSTER="abc"
PUBLISH_DIR=""
ARCHIVE_OUT=""

SKIP_MAPPING_REGEN=0
SKIP_CSV_REGEN=0
SKIP_FRESH_SANDBOX=1   # default: reuse existing $SANDBOX_DIR (faster iteration)
SKIP_PURGE=0
SKIP_INGEST=0
SKIP_SHUTDOWN=0
SKIP_PACK=0
SKIP_PUBLISH=0

usage() {
  cat <<'USAGE'
rebuild_synthetic_sandbox.sh - rebuild the local sandbox with synthetic data

Required:
  --synth-root PATH             Synthetic data root (parent of <DATE>/<fish>/<roi>/)

Channel mapping (one of):
  --channel-mapping PATH        Use this existing channel_mapping.json
  --membrane-fp-name NAME \     ... or regenerate channel_mapping.json from
  --fp-expressions-csv PATH         a *_cytosol_fp_expressions.csv

Common:
  --server-folder PATH          prepared.server_folder (default: --synth-root)
  --csv-out-dir   PATH          where prepared CSVs land (default: ${SCRATCH_ROOT}/synth_csvs)
  --cluster NAME                exists_<NAME>=true on each new prepared row (default: abc)
  --publish-dir PATH            copy final tarball here (default: $DATABASE_DIR from .env)
  --archive-out PATH            tarball output path (default: ${SCRATCH_ROOT}/${TODAY}_sandbox.tar.zst)

Skip flags (use to resume after a failure):
  --skip-mapping-regen          don't run generate_channel_mapping.py
  --skip-csv-regen              don't run build_synthetic_prepared_csvs.py
  --fresh-sandbox               re-extract the snapshot tarball (default: reuse)
  --skip-purge                  don't delete existing is_synthetic rows
  --skip-ingest                 don't run add_synthetic_metadata_to_db.py
  --skip-shutdown               leave sandbox running (skips pack + publish too)
  --skip-pack                   don't tar the sandbox
  --skip-publish                don't copy tarball to --publish-dir
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --synth-root)         SYNTH_ROOT="$2"; shift 2 ;;
    --server-folder)      SERVER_FOLDER="$2"; shift 2 ;;
    --membrane-fp-name)   MEMBRANE_FP_NAME="$2"; shift 2 ;;
    --fp-expressions-csv) FP_EXPRESSIONS_CSV="$2"; shift 2 ;;
    --channel-mapping)    CHANNEL_MAPPING="$2"; shift 2 ;;
    --csv-out-dir)        CSV_OUT_DIR="$2"; shift 2 ;;
    --cluster)            CLUSTER="$2"; shift 2 ;;
    --publish-dir)        PUBLISH_DIR="$2"; shift 2 ;;
    --archive-out)        ARCHIVE_OUT="$2"; shift 2 ;;
    --skip-mapping-regen) SKIP_MAPPING_REGEN=1; shift ;;
    --skip-csv-regen)     SKIP_CSV_REGEN=1; shift ;;
    --fresh-sandbox)      SKIP_FRESH_SANDBOX=0; shift ;;
    --skip-fresh-sandbox) SKIP_FRESH_SANDBOX=1; shift ;;   # documented for symmetry
    --skip-purge)         SKIP_PURGE=1; shift ;;
    --skip-ingest)        SKIP_INGEST=1; shift ;;
    --skip-shutdown)      SKIP_SHUTDOWN=1; SKIP_PACK=1; SKIP_PUBLISH=1; shift ;;
    --skip-pack)          SKIP_PACK=1; SKIP_PUBLISH=1; shift ;;
    --skip-publish)       SKIP_PUBLISH=1; shift ;;
    -h|--help)            usage; exit 0 ;;
    *)                    echo "[rebuild] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ env
if [[ -f "$ENV_FILE" ]]; then
  echo "[rebuild] sourcing $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[rebuild] WARN: $ENV_FILE not found; relying on environment" >&2
fi

# Resolve the host IP that postgres binds on. start_local_sandbox.sh exports
# it on stdout but not into our env; recompute the same way it does.
resolve_node_ip() {
  local raw_ip
  raw_ip="$(hostname --ip-address 2>/dev/null || true)"
  raw_ip="${raw_ip%% *}"
  [[ -n "$raw_ip" ]] && printf '%s' "$raw_ip"
}

: "${SUPABASE_LOCAL_PORT:?must be set in $ENV_FILE}"
: "${NODE_LOCAL_STORE_ROOT:?must be set in $ENV_FILE}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
SANDBOX_DIR="${SANDBOX_DIR:-${SCRATCH_ROOT}/sandbox}"
INSTANCE_NAME="${SANDBOX_INSTANCE_NAME:-sandbox_pg}"
SUPABASE_LOCAL_HOST="${SUPABASE_LOCAL_HOST:-$(resolve_node_ip)}"
SUPABASE_LOCAL_URI="postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres"
export SUPABASE_LOCAL_HOST SUPABASE_LOCAL_URI

# ------------------------------------------------------------------ resolve defaults / validate
[[ -n "$SYNTH_ROOT" ]] || { echo "[rebuild] --synth-root is required" >&2; usage; exit 2; }
[[ -d "$SYNTH_ROOT" ]] || { echo "[rebuild] --synth-root not a directory: $SYNTH_ROOT" >&2; exit 2; }
SERVER_FOLDER="${SERVER_FOLDER:-$SYNTH_ROOT}"
CSV_OUT_DIR="${CSV_OUT_DIR:-${SCRATCH_ROOT}/synth_csvs}"
PUBLISH_DIR="${PUBLISH_DIR:-${DATABASE_DIR:-}}"
TODAY=$(date +%Y_%m_%d)
ARCHIVE_OUT="${ARCHIVE_OUT:-${SCRATCH_ROOT}/${TODAY}_sandbox.tar.zst}"

mkdir -p "$CSV_OUT_DIR"

# Default channel_mapping path lives next to the CSVs unless caller pinned one
CHANNEL_MAPPING="${CHANNEL_MAPPING:-${CSV_OUT_DIR}/channel_mapping.json}"

# ------------------------------------------------------------------ tooling sanity
for cmd in psql tar zstd python apptainer rsync; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[rebuild] required tool not in PATH: $cmd" >&2; exit 1; }
done

# ============================================================ STEP 1: channel_mapping
if [[ "$SKIP_MAPPING_REGEN" -eq 0 ]]; then
  if [[ -z "$MEMBRANE_FP_NAME" || -z "$FP_EXPRESSIONS_CSV" ]]; then
    echo "[rebuild] --membrane-fp-name and --fp-expressions-csv are required to regenerate channel_mapping.json" >&2
    echo "          (or pass --skip-mapping-regen to reuse $CHANNEL_MAPPING)" >&2
    exit 2
  fi
  echo "[rebuild] STEP 1: regenerating $CHANNEL_MAPPING"
  mkdir -p "$(dirname "$CHANNEL_MAPPING")"
  python "${ML_DIR}/generate_channel_mapping.py" \
    --membrane-fp-name   "$MEMBRANE_FP_NAME" \
    --fp-expressions-csv "$FP_EXPRESSIONS_CSV" \
    --output             "$CHANNEL_MAPPING"
else
  echo "[rebuild] STEP 1: skipped channel_mapping regen (using $CHANNEL_MAPPING)"
fi
[[ -f "$CHANNEL_MAPPING" ]] || { echo "[rebuild] channel_mapping.json missing: $CHANNEL_MAPPING" >&2; exit 1; }

# ============================================================ STEP 2: prepared CSVs
if [[ "$SKIP_CSV_REGEN" -eq 0 ]]; then
  echo "[rebuild] STEP 2: regenerating prepared CSVs in $CSV_OUT_DIR"
  python "${ML_DIR}/build_synthetic_prepared_csvs.py" \
    --root            "$SYNTH_ROOT" \
    --server-folder   "$SERVER_FOLDER" \
    --channel-mapping "$CHANNEL_MAPPING" \
    --out-dir         "$CSV_OUT_DIR"
else
  echo "[rebuild] STEP 2: skipped CSV regen (using $CSV_OUT_DIR)"
fi
for f in prepared.csv prepared_tiles.csv prepared_cubes.csv prepared_annotations_3d.csv; do
  [[ -f "$CSV_OUT_DIR/$f" ]] || { echo "[rebuild] missing $CSV_OUT_DIR/$f" >&2; exit 1; }
done

# ============================================================ STEP 3: optional fresh extract
if [[ "$SKIP_FRESH_SANDBOX" -eq 0 ]]; then
  echo "[rebuild] STEP 3: --fresh-sandbox -> wiping $SANDBOX_DIR + tarball copy"
  "${SCRIPT_DIR}/stop_local_sandbox.sh" --clean
fi

# ============================================================ STEP 4: bring up sandbox
echo "[rebuild] STEP 4: starting sandbox"
SANDBOX_WAIT_SECONDS="${SANDBOX_WAIT_SECONDS:-300}" \
  "${SCRIPT_DIR}/start_local_sandbox.sh"

# Sanity: the start script prints "ready" but exits 0 either way; verify.
if ! psql "$SUPABASE_LOCAL_URI" -P pager=off -c "SELECT 1;" >/dev/null 2>&1; then
  echo "[rebuild] sandbox not reachable at $SUPABASE_LOCAL_URI" >&2
  echo "          see /tmp/${USER}/postgres.log for postgres logs" >&2
  exit 1
fi

# ============================================================ STEP 5: purge stale synthetic rows
if [[ "$SKIP_PURGE" -eq 0 ]]; then
  echo "[rebuild] STEP 5: purging existing is_synthetic rows + cache aggregates"

  # Capture target IDs once. We need them across multiple connections /
  # transactions because the cache aggregate tables are partitioned and
  # touching every partition in one transaction blows past the default
  # max_locks_per_transaction=64 ("out of shared memory").
  SYNTH_IDS="$(psql "$SUPABASE_LOCAL_URI" -At -v ON_ERROR_STOP=1 \
    -c "SELECT id FROM prepared WHERE COALESCE(is_synthetic, false) = true ORDER BY id")"

  if [[ -z "$SYNTH_IDS" ]]; then
    echo "[rebuild]   no existing synthetic rows to purge"
  else
    N_SYNTH=$(echo "$SYNTH_IDS" | wc -l)
    IDS_CSV=$(echo "$SYNTH_IDS" | paste -sd, -)
    echo "[rebuild]   purging $N_SYNTH ROIs (ids: $(echo "$SYNTH_IDS" | head -3 | tr '\n' ',' | sed 's/,$//')...)"

    # Phase A: enumerate cache aggregate tables and DELETE one at a time.
    # Each DELETE is its own implicit transaction so per-tx lock count
    # stays small (single relation + its partitions).
    CACHE_TABLES="$(psql "$SUPABASE_LOCAL_URI" -At -v ON_ERROR_STOP=1 -c "
      SELECT table_name FROM information_schema.tables
       WHERE table_schema = 'public'
         AND (   table_name LIKE 'prepared_cube_channel_agg_%'
              OR table_name LIKE 'prepared_cube_annotation_agg_%'
              OR table_name LIKE 'prepared_tile_channel_agg_%'
              OR table_name LIKE 'prepared_tile_annotation_agg_%')
       ORDER BY table_name")"
    N_CACHE=0
    if [[ -n "$CACHE_TABLES" ]]; then
      N_CACHE=$(echo "$CACHE_TABLES" | wc -l)
    fi
    echo "[rebuild]   draining $N_CACHE cache aggregate tables (autocommit)"
    while IFS= read -r tbl; do
      [[ -z "$tbl" ]] && continue
      psql "$SUPABASE_LOCAL_URI" -At -v ON_ERROR_STOP=1 \
        -c "DELETE FROM public.\"$tbl\" WHERE prepared_id IN ($IDS_CSV);" \
        >/dev/null
    done <<< "$CACHE_TABLES"

    # Phase A2: annotations_3d. The base table is heavily partitioned by
    # roi_id (partitions cover ~5 ids each), so a DELETE WHERE roi_id IN
    # ($IDS_CSV) only locks the partitions that match -- but we still
    # run it on its own implicit transaction to keep the lock budget low.
    echo "[rebuild]   draining annotations_3d for synthetic ROIs (autocommit)"
    psql "$SUPABASE_LOCAL_URI" -At -v ON_ERROR_STOP=1 \
      -c "DELETE FROM public.annotations_3d WHERE roi_id IN ($IDS_CSV);" \
      >/dev/null

    # Phase B: source tables in a single small transaction.
    # Order matters if FKs aren't set to CASCADE; explicit is cheap.
    echo "[rebuild]   deleting source rows (single tx)"
    psql "$SUPABASE_LOCAL_URI" -P pager=off -v ON_ERROR_STOP=1 <<SQL
BEGIN;
DELETE FROM prepared_tiles_view_table WHERE prepared_id IN ($IDS_CSV);
DELETE FROM prepared_cubes            WHERE prepared_id IN ($IDS_CSV);
DELETE FROM prepared_tiles            WHERE prepared_id IN ($IDS_CSV);
DELETE FROM prepared                  WHERE id            IN ($IDS_CSV);
COMMIT;
SQL
  fi
else
  echo "[rebuild] STEP 5: skipped purge"
fi

# ============================================================ STEP 6: ingest
if [[ "$SKIP_INGEST" -eq 0 ]]; then
  echo "[rebuild] STEP 6: ingesting prepared CSVs into local sandbox"
  python "${ML_DIR}/add_synthetic_metadata_to_db.py" \
    --db            local \
    --metadata-file "${CSV_OUT_DIR}/prepared.csv" \
    --cluster       "$CLUSTER"
else
  echo "[rebuild] STEP 6: skipped ingest"
fi

# ============================================================ STEP 6b: cross-table referential check
# Catches the failure mode the user explicitly called out: annotation
# rows landing under a roi_id that has no matching prepared_tiles row,
# or a tile_id that doesn't exist on the parent ROI. Either of those
# would make aggregate_prepared_tile_annotation_agg_1 silently produce
# zero rows for that ROI even though annotations_3d is non-empty.
echo "[rebuild] STEP 6b: cross-table consistency check"
psql "$SUPABASE_LOCAL_URI" -P pager=off -v ON_ERROR_STOP=1 <<'SQL'
\echo === orphan annotations: roi_id has no prepared row ===
SELECT a.roi_id, count(*) AS n_orphan
  FROM public.annotations_3d a
  LEFT JOIN public.prepared p ON p.id = a.roi_id
 WHERE a.is_consensus = true
   AND p.id IS NULL
 GROUP BY a.roi_id
 ORDER BY a.roi_id;

\echo === orphan annotations: (roi_id, tile_id) has no prepared_tiles row ===
SELECT a.roi_id, a.tile_id, count(*) AS n_orphan
  FROM public.annotations_3d a
  JOIN public.prepared p ON p.id = a.roi_id
  LEFT JOIN public.prepared_tiles pt
    ON pt.prepared_id = a.roi_id AND pt.tile_name = a.tile_id
 WHERE p.is_synthetic
   AND a.is_consensus = true
   AND pt.prepared_id IS NULL
 GROUP BY a.roi_id, a.tile_id
 ORDER BY a.roi_id, a.tile_id;

\echo === per-ROI summary: annotations vs tiles ===
SELECT p.id AS prepared_id,
       (SELECT count(*) FROM public.prepared_tiles  pt
         WHERE pt.prepared_id = p.id) AS n_tiles,
       (SELECT count(DISTINCT a.tile_id) FROM public.annotations_3d a
         WHERE a.roi_id = p.id AND a.is_consensus = true) AS n_anno_tiles,
       (SELECT count(*) FROM public.annotations_3d a
         WHERE a.roi_id = p.id AND a.is_consensus = true) AS n_annotations
  FROM public.prepared p
 WHERE p.is_synthetic
 ORDER BY p.id
 LIMIT 10;
SQL

# ============================================================ STEP 7: ANALYZE + CHECKPOINT
echo "[rebuild] STEP 7: ANALYZE + CHECKPOINT"
psql "$SUPABASE_LOCAL_URI" -P pager=off -c "ANALYZE;"
psql "$SUPABASE_LOCAL_URI" -P pager=off -c "CHECKPOINT;"

# ============================================================ STEP 8: verification
VERIF_OUT="${CSV_OUT_DIR}/verification_$(date +%Y%m%d_%H%M%S).txt"
echo "[rebuild] STEP 8: verification SQL -> $VERIF_OUT"
psql "$SUPABASE_LOCAL_URI" -P pager=off > "$VERIF_OUT" 2>&1 <<'SQL'
\echo === counts ===
SELECT
  (SELECT count(*) FROM prepared WHERE is_synthetic) AS synth_prepared,
  (SELECT count(*) FROM prepared_tiles_view_table ptv
     JOIN prepared p ON p.id = ptv.prepared_id WHERE p.is_synthetic) AS view_rows;

\echo === synthetic prepared sample ===
SELECT id, output_folder, time_size, channel_size, cube_size, is_synthetic,
       (channel_mapping::jsonb)
  FROM prepared
 WHERE is_synthetic
 ORDER BY id DESC
 LIMIT 5;

\echo === cube counts per (prepared_id, channel) ===
SELECT pc.prepared_id, pc.channel, count(*) AS n_cubes
  FROM prepared_cubes pc
  JOIN prepared p ON p.id = pc.prepared_id
 WHERE p.is_synthetic
 GROUP BY pc.prepared_id, pc.channel
 ORDER BY pc.prepared_id DESC, pc.channel
 LIMIT 25;

\echo === annotations_3d totals for synthetic ROIs ===
SELECT count(*)              AS n_annotations,
       count(DISTINCT roi_id) AS n_rois,
       count(DISTINCT (roi_id, tile_id)) AS n_tiles
  FROM annotations_3d a
  JOIN prepared p ON p.id = a.roi_id
 WHERE p.is_synthetic
   AND a.is_consensus = true;

\echo === prepared_tile_annotation_agg_1 rolled up for synthetic ROIs ===
SELECT count(*)                       AS n_agg_rows,
       sum(annotation_count)::bigint  AS sum_annotation_count,
       count(DISTINCT prepared_id)    AS n_rois_with_agg
  FROM prepared_tile_annotation_agg_1 a
  JOIN prepared p ON p.id = a.prepared_id
 WHERE p.is_synthetic;
SQL
sed -n '1,60p' "$VERIF_OUT"

# ============================================================ STEP 9: clean shutdown
if [[ "$SKIP_SHUTDOWN" -eq 1 ]]; then
  echo "[rebuild] STEP 9: skipped shutdown -- sandbox left running for inspection"
  echo "[rebuild] done (partial)"
  exit 0
fi

echo "[rebuild] STEP 9: clean shutdown"
"${SCRIPT_DIR}/stop_local_sandbox.sh"

PID_FILE="$SANDBOX_DIR/var/lib/postgresql/data/postmaster.pid"
SOCK_GLOB="$SANDBOX_DIR/run/postgresql/.s.PGSQL.${SUPABASE_LOCAL_PORT}*"
if [[ -e "$PID_FILE" ]]; then
  echo "[rebuild] FATAL: $PID_FILE still present after shutdown -- not packing" >&2
  exit 1
fi
# shellcheck disable=SC2086
if compgen -G "$SOCK_GLOB" >/dev/null; then
  echo "[rebuild] FATAL: stale socket lock(s) present after shutdown:" >&2
  ls -la $SOCK_GLOB >&2 || true
  exit 1
fi

# ============================================================ STEP 10: pack
if [[ "$SKIP_PACK" -eq 1 ]]; then
  echo "[rebuild] STEP 10: skipped pack"
else
  echo "[rebuild] STEP 10: packing $SANDBOX_DIR -> $ARCHIVE_OUT"
  du -sh "$SANDBOX_DIR" || true
  df -h "$(dirname "$ARCHIVE_OUT")" || true
  time tar -I 'zstd -3 -T0' -cf "$ARCHIVE_OUT" -C "$SCRATCH_ROOT" sandbox
  ls -lh "$ARCHIVE_OUT"
fi

# ============================================================ STEP 11: publish
if [[ "$SKIP_PUBLISH" -eq 1 ]]; then
  echo "[rebuild] STEP 11: skipped publish"
else
  if [[ -z "$PUBLISH_DIR" ]]; then
    echo "[rebuild] STEP 11: no --publish-dir / DATABASE_DIR set; skipping publish"
  else
    DEST="${PUBLISH_DIR}/${TODAY}_sandbox.tar.zst"
    TMP="${DEST}.inflight"
    echo "[rebuild] STEP 11: publishing -> $DEST"
    mkdir -p "$PUBLISH_DIR"
    cp -v "$ARCHIVE_OUT" "$TMP"
    mv -v "$TMP" "$DEST"
    ls -lh "$DEST"
    cat <<EOF

[rebuild] update DATABASE_SANDBOX in $ENV_FILE to point at:
  DATABASE_SANDBOX=${DEST}
EOF
  fi
fi

echo "[rebuild] done"

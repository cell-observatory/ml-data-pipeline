#!/usr/bin/env bash
# Tight follow-up introspection: dump only what we still need to know about
# the annotation pipeline.
#
#   1. body of aggregate_prepared_tile_annotation_agg_1 (the real worker
#      that builds annotations_metadata + annotation_count for the agg)
#   2. column list of public.annotations_3d (the source table we have to
#      populate from synthetic *_staging_annotations_3d.csv files)
#   3. body of aggregate_prepared_cube_annotation_aggs_from_channel_cache
#      (sibling for cube aggs; same logic at cube-level if needed later)
#   4. one real annotations_metadata sample from a non-synthetic ROI in the
#      agg (locks down the exact JSON shape per element)
#
# Output is small (KBs, not GBs). Re-run after start_local_sandbox.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/clusterfs/nvme/martinalvarez/GitHub/cell_observatory_platform}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

resolve_node_ip() {
  local raw_ip
  raw_ip="$(hostname --ip-address 2>/dev/null || true)"
  raw_ip="${raw_ip%% *}"
  [[ -n "$raw_ip" ]] && printf '%s' "$raw_ip"
}

: "${SUPABASE_LOCAL_PORT:?must be set in $ENV_FILE}"
SUPABASE_LOCAL_HOST="${SUPABASE_LOCAL_HOST:-$(resolve_node_ip)}"
SUPABASE_LOCAL_URI="${SUPABASE_LOCAL_URI:-postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres}"

if ! psql "$SUPABASE_LOCAL_URI" -P pager=off -c 'SELECT 1' >/dev/null 2>&1; then
  echo "[introspect] sandbox not reachable at $SUPABASE_LOCAL_URI" >&2
  exit 1
fi

psql "$SUPABASE_LOCAL_URI" -P pager=off -X <<'SQL'
\echo === (A) aggregate_prepared_tile_annotation_agg_1 body ===
SELECT pg_get_functiondef(p.oid)
  FROM pg_proc p
  JOIN pg_namespace n ON n.oid = p.pronamespace
 WHERE p.proname = 'aggregate_prepared_tile_annotation_agg_1'
   AND n.nspname = 'public'
 LIMIT 1;

\echo
\echo === (B) annotations_3d full column list ===
SELECT column_name, data_type, is_nullable, column_default
  FROM information_schema.columns
 WHERE table_schema = 'public' AND table_name = 'annotations_3d'
 ORDER BY ordinal_position;

\echo
\echo === (B2) annotations_3d primary/unique keys ===
SELECT tc.constraint_name, tc.constraint_type, kcu.column_name, kcu.ordinal_position
  FROM information_schema.table_constraints tc
  JOIN information_schema.key_column_usage kcu
    ON tc.constraint_name = kcu.constraint_name
   AND tc.table_schema    = kcu.table_schema
 WHERE tc.table_schema = 'public'
   AND tc.table_name   = 'annotations_3d'
   AND tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
 ORDER BY tc.constraint_name, kcu.ordinal_position;

\echo
\echo === (C) aggregate_prepared_cube_annotation_aggs_from_channel_cache body (for context) ===
SELECT pg_get_functiondef(p.oid)
  FROM pg_proc p
  JOIN pg_namespace n ON n.oid = p.pronamespace
 WHERE p.proname = 'aggregate_prepared_cube_annotation_aggs_from_channel_cache'
   AND n.nspname = 'public'
 LIMIT 1;

\echo
\echo === (D) one real annotations_metadata sample from prepared_tile_annotation_agg_1 ===
SELECT a.prepared_id,
       a.tile_name,
       a.annotation_count,
       jsonb_array_length(a.annotations_metadata) AS n_annotations,
       jsonb_pretty( a.annotations_metadata -> 0 ) AS first_annotation
  FROM public.prepared_tile_annotation_agg_1 a
  JOIN public.prepared p ON p.id = a.prepared_id
 WHERE COALESCE(p.is_synthetic, false) = false
   AND a.annotations_metadata IS NOT NULL
   AND jsonb_typeof(a.annotations_metadata) = 'array'
   AND jsonb_array_length(a.annotations_metadata) > 0
 ORDER BY a.prepared_id DESC
 LIMIT 1;

\echo
\echo === (D2) keys present across all annotation objects in that sample ===
WITH one_row AS (
  SELECT a.annotations_metadata AS am
    FROM public.prepared_tile_annotation_agg_1 a
    JOIN public.prepared p ON p.id = a.prepared_id
   WHERE COALESCE(p.is_synthetic, false) = false
     AND a.annotations_metadata IS NOT NULL
     AND jsonb_typeof(a.annotations_metadata) = 'array'
     AND jsonb_array_length(a.annotations_metadata) > 0
   ORDER BY a.prepared_id DESC LIMIT 1
)
SELECT DISTINCT k
  FROM one_row,
       LATERAL jsonb_array_elements(am) elem,
       LATERAL jsonb_object_keys(elem) k
 ORDER BY k;
SQL

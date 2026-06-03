#!/usr/bin/env bash
# Host-runnable wrapper around cell_observatory_platform's
# scripts/db/local_start_sandbox.sh.
#
# The upstream script hardcodes `cd /workspace/cell_observatory_platform`
# (it's meant to run inside the dev container). This wrapper sources the
# repo .env from $REPO_DIR (or its default path) and otherwise mirrors the
# upstream behavior: copy the snapshot tarball to local scratch, extract
# it, start an Apptainer Postgres instance on $SUPABASE_LOCAL_PORT, and
# wait until psql can reach it.
#
# Required env (sourced from .env):
#   DATABASE_SANDBOX        path to the sandbox.tar.zst snapshot
#   SUPABASE_LOCAL_PORT     port to listen on (e.g. 54322)
#   NODE_LOCAL_STORE_ROOT   used to derive SCRATCH_ROOT if unset
# Optional:
#   SCRATCH_ROOT            where to copy + extract (default: dirname NODE_LOCAL_STORE_ROOT)
#   SANDBOX_DIR             extracted sandbox path (default: $SCRATCH_ROOT/sandbox)
#   SANDBOX_INSTANCE_NAME   apptainer instance name (default: sandbox_pg)
#   SANDBOX_WAIT_SECONDS    psql readiness timeout (default: 60)
#   SUPABASE_LOCAL_HOST     bind/advertise host (default: hostname --ip-address)
#   REPO_DIR / ENV_FILE     override the .env source location

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clusterfs/nvme/martinalvarez/GitHub/cell_observatory_platform}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [[ -f "$ENV_FILE" ]]; then
  echo "[start_local_sandbox] sourcing $ENV_FILE"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
else
  echo "[start_local_sandbox] WARN: $ENV_FILE not found; relying on environment"
fi

: "${DATABASE_SANDBOX:?must be set (e.g. via $ENV_FILE)}"
: "${SUPABASE_LOCAL_PORT:?must be set}"
: "${NODE_LOCAL_STORE_ROOT:?must be set}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
SANDBOX_TAR="${SCRATCH_ROOT}/sandbox.tar.zst"
SANDBOX_DIR="${SANDBOX_DIR:-${SCRATCH_ROOT}/sandbox}"
INSTANCE_NAME="${SANDBOX_INSTANCE_NAME:-sandbox_pg}"
WAIT_SECONDS="${SANDBOX_WAIT_SECONDS:-60}"

resolve_node_ip() {
  local raw_ip
  raw_ip="$(hostname --ip-address 2>/dev/null || true)"
  raw_ip="${raw_ip%% *}"
  [[ -n "$raw_ip" ]] && printf '%s' "$raw_ip"
}

SUPABASE_LOCAL_HOST="${SUPABASE_LOCAL_HOST:-$(resolve_node_ip)}"
if [[ -z "${SUPABASE_LOCAL_HOST}" ]]; then
  echo "[start_local_sandbox] could not resolve SUPABASE_LOCAL_HOST"
  exit 1
fi
SUPABASE_LOCAL_URI="postgresql://postgres:postgres@${SUPABASE_LOCAL_HOST}:${SUPABASE_LOCAL_PORT}/postgres"
export SUPABASE_LOCAL_HOST SUPABASE_LOCAL_URI

for cmd in apptainer psql rsync tar zstd; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "[start_local_sandbox] $cmd not found in PATH"; exit 1; }
done

mkdir -p "$SCRATCH_ROOT" "$NODE_LOCAL_STORE_ROOT"

echo "[start_local_sandbox] stopping any existing instance: $INSTANCE_NAME"
apptainer instance stop "$INSTANCE_NAME" >/dev/null 2>&1 || true

if [[ -d "$SANDBOX_DIR" ]]; then
  echo "[start_local_sandbox] reusing existing $SANDBOX_DIR"
  echo "                       (delete it manually for a clean re-extract)"
else
  echo "[start_local_sandbox] copying tarball to $SANDBOX_TAR"
  rsync -av --progress "$DATABASE_SANDBOX" "$SANDBOX_TAR"
  echo "[start_local_sandbox] extracting under $SCRATCH_ROOT"
  zstd -d -c "$SANDBOX_TAR" | tar -xf - -C "$SCRATCH_ROOT"
  if [[ ! -d "$SANDBOX_DIR" ]]; then
    echo "[start_local_sandbox] expected extracted sandbox at $SANDBOX_DIR but not found"
    exit 1
  fi
fi

# Pre-create bind-mount targets inside the writable sandbox so apptainer's
# auto-bind doesn't fail when these paths exist on the host.
for dest in "${DATA_DIR:-}" "${STORAGE_SERVER_DIR:-}" "${DATABASE_DIR:-}" \
            "$NODE_LOCAL_STORE_ROOT" "/scratch" "/dev/shm"; do
  [[ -n "$dest" ]] && mkdir -p "${SANDBOX_DIR}${dest}"
done

echo "[start_local_sandbox] starting Postgres apptainer instance: $INSTANCE_NAME"
env -u APPTAINER_BIND \
    -u APPTAINER_BINDPATH \
    -u SINGULARITY_BIND \
    -u SINGULARITY_BINDPATH \
    apptainer instance start \
      --no-mount proc \
      --writable \
      --env POSTGRES_PASSWORD=postgres \
      "$SANDBOX_DIR" \
      "$INSTANCE_NAME"

# Pre-clean stale runtime artifacts left by a prior unclean shutdown.
# Postgres refuses to start if these exist and the PID inside postmaster.pid
# is reused by the host kernel, or if the unix-socket lock file is stale.
# We only delete after confirming the claimed PID is dead -- never blindly
# remove postmaster.pid for a live process.
PID_FILE="$SANDBOX_DIR/var/lib/postgresql/data/postmaster.pid"
if [[ -e "$PID_FILE" ]]; then
  CLAIMED_PID=$(awk 'NR==1{print $1}' "$PID_FILE" 2>/dev/null || true)
  if [[ -n "$CLAIMED_PID" ]] && ! kill -0 "$CLAIMED_PID" 2>/dev/null; then
    echo "[start_local_sandbox] removing stale $PID_FILE (pid $CLAIMED_PID is dead)"
    rm -f "$PID_FILE"
  else
    echo "[start_local_sandbox] WARN: $PID_FILE present and pid $CLAIMED_PID may be live; leaving in place"
  fi
fi
# Unix-socket lock files are always safe to remove between instance restarts:
# the previous in-container postgres is gone after `instance stop`.
SOCK_GLOB="$SANDBOX_DIR/run/postgresql/.s.PGSQL.${SUPABASE_LOCAL_PORT}*"
# shellcheck disable=SC2086
if compgen -G "$SOCK_GLOB" >/dev/null; then
  echo "[start_local_sandbox] removing stale socket lock(s): $SOCK_GLOB"
  # shellcheck disable=SC2086
  rm -f $SOCK_GLOB
fi

echo "[start_local_sandbox] launching postgres inside instance"
apptainer exec instance://"$INSTANCE_NAME" postgres \
    -c "listen_addresses=0.0.0.0" \
    -c "port=${SUPABASE_LOCAL_PORT}" \
    -c 'config_file=/etc/postgresql/postgresql.conf' \
    >"${SCRATCH_ROOT}/postgres.log" 2>&1 &

echo "[start_local_sandbox] waiting for $SUPABASE_LOCAL_URI"
ready=0
for ((i=0; i<WAIT_SECONDS; i+=2)); do
  if psql "$SUPABASE_LOCAL_URI" --command="SELECT 1;" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 2
done

if [[ "$ready" -ne 1 ]]; then
  echo "[start_local_sandbox] postgres did not become ready in ${WAIT_SECONDS}s"
  echo "                       see ${SCRATCH_ROOT}/postgres.log for details"
  apptainer instance list || true
  exit 1
fi

echo "[start_local_sandbox] ready"
psql "$SUPABASE_LOCAL_URI" --command="SELECT current_database(), current_user, version();"

cat <<EOF

To use with the synthetic-data ingest from a fresh shell:
  export SUPABASE_LOCAL_HOST=${SUPABASE_LOCAL_HOST}
  export SUPABASE_LOCAL_URI="${SUPABASE_LOCAL_URI}"

EOF

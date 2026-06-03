#!/usr/bin/env bash
# Host-runnable counterpart to start_local_sandbox.sh. Performs a graceful
# Postgres shutdown from inside the apptainer instance (so postmaster.pid
# is removed and any future restart can recover without manual cleanup),
# then stops the apptainer instance. Pass --clean to also remove the
# extracted sandbox dir and the copied tarball from $SCRATCH_ROOT.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/clusterfs/nvme/martinalvarez/GitHub/cell_observatory_platform}"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

: "${NODE_LOCAL_STORE_ROOT:?must be set}"

SCRATCH_ROOT="${SCRATCH_ROOT:-$(dirname "$NODE_LOCAL_STORE_ROOT")}"
SANDBOX_DIR="${SANDBOX_DIR:-${SCRATCH_ROOT}/sandbox}"
SANDBOX_TAR="${SCRATCH_ROOT}/sandbox.tar.zst"
INSTANCE_NAME="${SANDBOX_INSTANCE_NAME:-sandbox_pg}"

# Graceful shutdown only if the instance is actually running. `apptainer
# exec` against a non-existent instance prints a noisy error, so probe
# the instance list first.
if apptainer instance list 2>/dev/null | awk 'NR>1 {print $1}' | grep -qx "$INSTANCE_NAME"; then
  echo "[stop_local_sandbox] graceful pg_ctl stop inside $INSTANCE_NAME"
  if ! apptainer exec "instance://$INSTANCE_NAME" \
        pg_ctl -D /var/lib/postgresql/data -m fast -w stop 2>&1 \
        | sed 's/^/  /'; then
    # postgres may already be down; still proceed to instance stop
    echo "[stop_local_sandbox] WARN: pg_ctl stop returned non-zero; proceeding"
  fi
fi

echo "[stop_local_sandbox] stopping instance: $INSTANCE_NAME"
apptainer instance stop "$INSTANCE_NAME" >/dev/null 2>&1 || true

if [[ "${1:-}" == "--clean" ]]; then
  echo "[stop_local_sandbox] removing $SANDBOX_DIR"
  rm -rf "$SANDBOX_DIR"
  echo "[stop_local_sandbox] removing $SANDBOX_TAR"
  rm -f "$SANDBOX_TAR"
fi

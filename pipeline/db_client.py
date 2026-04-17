from __future__ import annotations

import os
import logging
import subprocess
import socket
import time
from pathlib import Path
from typing import Optional

import connectorx as cx
import pyarrow as pa
from dotenv import load_dotenv
import psycopg
from supabase import create_client
from supabase.lib.client_options import SyncClientOptions

logger = logging.getLogger(__name__)

_MODES = ("local", "staging", "prod")
_DEFAULT_DOTENV = Path(__file__).resolve().parent.parent / ".env"

_ENV_KEYS = {
    "local": "SUPABASE_LOCAL_PORT",
    "staging": "SUPABASE_STAGING_URI",
    "prod": "SUPABASE_PROD_URI",
}


def _resolve_local_pg_host() -> Optional[str]:
    explicit_host = os.environ.get("SUPABASE_LOCAL_HOST")
    if explicit_host:
        return explicit_host

    try:
        raw_ip = subprocess.check_output(
            ["hostname", "--ip-address"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if raw_ip:
            first_ip = raw_ip.split()[0]
            if first_ip and not first_ip.startswith("127."):
                return first_ip
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
        pass

    try:
        hostname = socket.gethostname()
        for _, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
            ip = sockaddr[0]
            if ip and not ip.startswith("127."):
                return ip
    except socket.gaierror:
        pass

    return None


class PipelineDBClient:
    """Thin wrapper around connectorx (reads) and mode-specific write clients.

    Modes
    -----
    local    - reads and writes against a local Postgres/Supabase sandbox.
    staging  - reads against the staging Postgres URI, writes via Supabase REST.
    prod     - reads against the production Postgres URI, writes via Supabase REST.
    """

    def __init__(
        self,
        mode: str = "prod",
        dotenv_path: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        statement_timeout_ms: int = 600_000,
        verbose: bool = True,
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {mode!r}")
        self.mode = mode
        self.verbose = verbose
        self.statement_timeout_ms = statement_timeout_ms

        self.dotenv_path = Path(dotenv_path) if dotenv_path is not None else _DEFAULT_DOTENV
        if dotenv_path is not None and not self.dotenv_path.exists():
            raise FileNotFoundError(f"dotenv_path={dotenv_path!r} does not exist")
        if self.dotenv_path.exists():
            load_dotenv(self.dotenv_path, verbose=verbose, override=False)

        self._local_pg_uri = self._build_local_pg_uri()
        self._read_uri = self._build_read_uri()
        self._write_backend = "sql" if self.mode == "local" else "supabase"
        self._write_uri = self._local_pg_uri if self.mode == "local" else None
        self._supabase_url = ""
        self._supabase_key = ""
        if self._write_backend == "supabase":
            self._supabase_url = supabase_url or os.environ.get("SUPABASE_URL", "")
            self._supabase_key = supabase_key or os.environ.get("SUPABASE_KEY", "")

        if verbose:
            logger.info(
                "[PipelineDBClient] mode=%s read_uri=%s write_backend=%s",
                self.mode,
                self._read_uri[:40] + "..." if len(self._read_uri) > 40 else self._read_uri,
                self._write_backend,
            )

    def _build_local_pg_uri(self) -> Optional[str]:
        explicit_uri = os.environ.get("SUPABASE_LOCAL_URI")
        if explicit_uri:
            return explicit_uri

        port = os.environ.get("SUPABASE_LOCAL_PORT")
        host = _resolve_local_pg_host()
        if host and port:
            return f"postgresql://postgres:postgres@{host}:{int(port)}/postgres"

        if self.mode == "local":
            raise ValueError(
                "mode='local' requires SUPABASE_LOCAL_URI or both "
                "SUPABASE_LOCAL_HOST and SUPABASE_LOCAL_PORT. "
                "As a fallback, local mode derives the host from the current "
                "node IP resolver and uses SUPABASE_LOCAL_PORT."
            )
        return None

    def _build_read_uri(self) -> str:
        if self.mode == "local":
            assert self._local_pg_uri is not None
            return self._local_pg_uri
        env_key = _ENV_KEYS[self.mode]
        uri = os.environ.get(env_key)
        if not uri:
            raise ValueError(f"{env_key} must be set for mode={self.mode!r}")
        return uri

    @property
    def read_uri(self) -> str:
        return self._read_uri

    @property
    def write_backend(self) -> str:
        return self._write_backend

    @property
    def write_uri(self) -> Optional[str]:
        return self._write_uri

    def query_arrow(self, sql: str) -> pa.Table:
        t0 = time.perf_counter()
        preview = " ".join(sql.split())[:160]
        if self.verbose:
            logger.info("[PipelineDBClient] query: %s", preview)
        table = cx.read_sql(
            conn=self._read_uri,
            query=sql,
            protocol="cursor",
            return_type="arrow",
            pre_execution_query=[f"SET statement_timeout = '{self.statement_timeout_ms}';"],
        )
        if self.verbose:
            logger.info(
                "[PipelineDBClient] rows=%s elapsed=%.2fs",
                table.num_rows,
                time.perf_counter() - t0,
            )
        return table

    def supabase_write_client(self):
        """Return a Supabase SyncClient for remote insert/update/rpc calls."""
        if self._write_backend != "supabase":
            raise ValueError("supabase_write_client() is only available for staging/prod modes")
        if not self._supabase_url or not self._supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be set for write operations. "
                "Pass them explicitly or set them in your .env file."
            )
        return create_client(
            self._supabase_url,
            self._supabase_key,
            options=SyncClientOptions(
                postgrest_client_timeout=600,
                storage_client_timeout=600,
                schema="public",
            ),
        )

    def pg_write_connection(self):
        """Return a psycopg connection for local SQL writes."""
        if self._write_backend != "sql" or not self._write_uri:
            raise ValueError("pg_write_connection() is only available for local mode")
        return psycopg.connect(self._write_uri)

    def rpc(self, fn_name: str, params: dict):
        """Call a Supabase RPC function (e.g. refresh_prepared_cache_artifacts)."""
        if self._write_backend != "supabase":
            raise ValueError("rpc() is only available for staging/prod modes")
        client = self.supabase_write_client()
        return client.rpc(fn_name, params).execute()

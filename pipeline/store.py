from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Optional

from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

logger = logging.getLogger(__name__)


class PipelineStore(ABC):
    """Abstract persistence backend for scan logs and prepared data."""

    @abstractmethod
    def write_scan_log(self, entries: list[dict]) -> None: ...

    @abstractmethod
    def read_scan_log(self, roi_acquisition_id: Optional[str] = None) -> list[dict]: ...

    @abstractmethod
    def read_scan_failures(self, since_days: int = 7) -> list[dict]: ...

    @abstractmethod
    def ingest_prepared_roi(
        self,
        prepared: dict,
        tiles: list[dict],
        cubes: list[dict],
    ) -> int:
        """Insert prepared + tiles + cubes. Returns prepared_id.

        Does NOT refresh aggregate caches. Callers should invoke
        ``refresh_cache_artifacts`` once after ingesting all ROIs.
        """
        ...

    @abstractmethod
    def refresh_cache_artifacts(self, prepared_ids: list[int]) -> None:
        """Refresh aggregate cache artifacts for the given prepared_ids.

        This is the expensive step and should run once per batch, not per ROI.
        """
        ...


class JsonFileStore(PipelineStore):
    """Writes structured JSON to disk. Useful for debugging and offline runs."""

    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir)
        self._scan_log_dir = self.base_dir / "scan_logs"
        self._prepared_dir = self.base_dir / "prepared"
        self._scan_log_dir.mkdir(parents=True, exist_ok=True)
        self._prepared_dir.mkdir(parents=True, exist_ok=True)

    def write_scan_log(self, entries: list[dict]) -> None:
        ts = int(time.time())
        path = self._scan_log_dir / f"scan_log_{ts}.json"
        path.write_text(json.dumps(entries, indent=2, default=str))
        logger.info("[JsonFileStore] wrote %d scan log entries to %s", len(entries), path)

    def read_scan_log(self, roi_acquisition_id: Optional[str] = None) -> list[dict]:
        all_entries: list[dict] = []
        for p in sorted(self._scan_log_dir.glob("scan_log_*.json")):
            with open(p) as f:
                all_entries.extend(json.load(f))
        if roi_acquisition_id:
            all_entries = [e for e in all_entries if e.get("roi_acquisition_id") == roi_acquisition_id]
        return all_entries

    def read_scan_failures(self, since_days: int = 7) -> list[dict]:
        cutoff = time.time() - since_days * 86400
        failures: list[dict] = []
        for p in sorted(self._scan_log_dir.glob("scan_log_*.json")):
            ts_str = p.stem.replace("scan_log_", "")
            try:
                file_ts = int(ts_str)
            except ValueError:
                continue
            if file_ts < cutoff:
                continue
            with open(p) as f:
                entries = json.load(f)
            failures.extend(e for e in entries if e.get("status") == "failed")
        return failures

    def ingest_prepared_roi(
        self,
        prepared: dict,
        tiles: list[dict],
        cubes: list[dict],
    ) -> int:
        roi_id = prepared.get("raw_roi_acquisition_id", "unknown")
        payload = {
            "prepared": prepared,
            "tiles": tiles,
            "cubes_count": len(cubes),
            "cubes_sample": cubes[:5],
        }
        path = self._prepared_dir / f"prepared_{roi_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("[JsonFileStore] wrote prepared roi %s to %s", roi_id, path)
        return 0

    def refresh_cache_artifacts(self, prepared_ids: list[int]) -> None:
        return None


class LocalPostgresStore(PipelineStore):
    """Writes directly to a local Postgres sandbox via SQL."""

    CUBE_INSERT_BATCH = 10_000
    _JSON_COLUMNS = {
        "roi_tile_scan_log": {"error_messages", "scan_metadata"},
        "prepared": {"channel_mapping"},
        "prepared_tiles": set(),
        "prepared_cubes": set(),
    }

    def __init__(self, db_client) -> None:
        from pipeline.db_client import PipelineDBClient
        self.db_client: PipelineDBClient = db_client

    def _connect(self):
        return self.db_client.pg_write_connection()

    def _adapt_value(self, table_name: str, column: str, value: Any) -> Any:
        if value is None:
            return None
        if column in self._JSON_COLUMNS.get(table_name, set()):
            if isinstance(value, str):
                return Jsonb(json.loads(value))
            return Jsonb(value)
        return value

    def _insert_row(
        self,
        cur,
        table_name: str,
        row: dict[str, Any],
        returning: Optional[str] = None,
    ):
        columns = list(row.keys())
        values = [self._adapt_value(table_name, col, row[col]) for col in columns]
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )
        if returning:
            query += sql.SQL(" RETURNING {}").format(sql.Identifier(returning))
        cur.execute(query, values)
        if returning:
            result = cur.fetchone()
            if result is None:
                raise ValueError(f"Insert into {table_name} did not return {returning}")
            return result[0]
        return None

    def _insert_many(self, cur, table_name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        columns = list(rows[0].keys())
        query = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            sql.Identifier(table_name),
            sql.SQL(", ").join(sql.Identifier(col) for col in columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in columns),
        )
        values = [
            [self._adapt_value(table_name, col, row[col]) for col in columns]
            for row in rows
        ]
        cur.executemany(query, values)

    def write_scan_log(self, entries: list[dict]) -> None:
        if not entries:
            return
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                self._insert_many(cur, "roi_tile_scan_log", entries)
            conn.commit()
            logger.info("[LocalPostgresStore] inserted %d scan log entries", len(entries))
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def read_scan_log(self, roi_acquisition_id: Optional[str] = None) -> list[dict]:
        conn = self._connect()
        try:
            query = "SELECT * FROM roi_tile_scan_log"
            params: list[Any] = []
            if roi_acquisition_id:
                query += " WHERE roi_acquisition_id = %s"
                params.append(roi_acquisition_id)
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, params)
                return list(cur.fetchall())
        finally:
            conn.close()

    def read_scan_failures(self, since_days: int = 7) -> list[dict]:
        conn = self._connect()
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM roi_tile_scan_log
                    WHERE status = 'failed'
                      AND scanned_at >= NOW() - (%s * INTERVAL '1 day')
                    """,
                    [since_days],
                )
                return list(cur.fetchall())
        finally:
            conn.close()

    def ingest_prepared_roi(
        self,
        prepared: dict,
        tiles: list[dict],
        cubes: list[dict],
    ) -> int:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                prepared_id = self._insert_row(cur, "prepared", prepared, returning="id")
                for tile in tiles:
                    tile["prepared_id"] = prepared_id
                self._insert_many(cur, "prepared_tiles", tiles)

                for cube in cubes:
                    cube["prepared_id"] = prepared_id
                for i in range(0, len(cubes), self.CUBE_INSERT_BATCH):
                    batch = cubes[i : i + self.CUBE_INSERT_BATCH]
                    self._insert_many(cur, "prepared_cubes", batch)
            conn.commit()
            logger.debug(
                "[LocalPostgresStore] ingested prepared_id=%d (%d tiles, %d cubes)",
                prepared_id,
                len(tiles),
                len(cubes),
            )
            return prepared_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def refresh_cache_artifacts(self, prepared_ids: list[int]) -> None:
        if not prepared_ids:
            return
        from tqdm import tqdm

        # ANALYZE the hot tables before refreshing caches. The ingest loop just
        # bulk-inserted ~O(100k) rows per ROI into prepared_cubes partitions;
        # without fresh stats the planner picks seq-scans over index-scans
        # inside refresh_prepared_cube_*_aggs, roughly doubling wall time.
        analyze_targets = [
            "public.prepared",
            "public.prepared_tiles",
            "public.prepared_tiles_view_table",
            "public.prepared_cubes",
        ]
        conn = self._connect()
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                for target in tqdm(
                    analyze_targets,
                    desc="Analyze hot tables",
                    unit="tbl",
                ):
                    cur.execute(f"ANALYZE {target}")
            conn.autocommit = False
            with conn.cursor() as cur:
                for prepared_id in tqdm(
                    prepared_ids,
                    desc="Refresh cache artifacts",
                    unit="roi",
                ):
                    cur.execute(
                        "SELECT refresh_prepared_cache_artifacts(%s)",
                        [prepared_id],
                    )
                    conn.commit()
            logger.info(
                "[LocalPostgresStore] refreshed cache artifacts for %d prepared ROIs",
                len(prepared_ids),
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


class SupabaseStore(PipelineStore):
    """Writes to Supabase via the API. Inserts always use the remote client."""

    CUBE_INSERT_BATCH = 10_000

    def __init__(self, db_client) -> None:
        from pipeline.db_client import PipelineDBClient
        self.db_client: PipelineDBClient = db_client
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = self.db_client.supabase_write_client()
        return self._client

    def write_scan_log(self, entries: list[dict]) -> None:
        if not entries:
            return
        self.client.table("roi_tile_scan_log").insert(entries).execute()
        logger.info("[SupabaseStore] inserted %d scan log entries", len(entries))

    def read_scan_log(self, roi_acquisition_id: Optional[str] = None) -> list[dict]:
        q = self.client.table("roi_tile_scan_log").select("*")
        if roi_acquisition_id:
            q = q.eq("roi_acquisition_id", roi_acquisition_id)
        return q.execute().data

    def read_scan_failures(self, since_days: int = 7) -> list[dict]:
        from datetime import datetime, timedelta, timezone
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        return (
            self.client.table("roi_tile_scan_log")
            .select("*")
            .eq("status", "failed")
            .gte("scanned_at", cutoff)
            .execute()
            .data
        )

    def ingest_prepared_roi(
        self,
        prepared: dict,
        tiles: list[dict],
        cubes: list[dict],
    ) -> int:
        prepared_id = None
        try:
            resp = self.client.table("prepared").insert(prepared).execute()
            prepared_id = resp.data[0]["id"]

            for tile in tiles:
                tile["prepared_id"] = prepared_id
            self.client.table("prepared_tiles").insert(tiles).execute()

            for cube in cubes:
                cube["prepared_id"] = prepared_id
            for i in range(0, len(cubes), self.CUBE_INSERT_BATCH):
                batch = cubes[i : i + self.CUBE_INSERT_BATCH]
                self.client.table("prepared_cubes").insert(batch).execute()
            logger.debug(
                "[SupabaseStore] ingested prepared_id=%d (%d tiles, %d cubes)",
                prepared_id,
                len(tiles),
                len(cubes),
            )
            return prepared_id

        except Exception:
            if prepared_id is not None:
                logger.warning("[SupabaseStore] rolling back prepared_id=%d", prepared_id)
                self.client.table("prepared").delete().eq("id", prepared_id).execute()
            raise

    def refresh_cache_artifacts(self, prepared_ids: list[int]) -> None:
        if not prepared_ids:
            return
        from tqdm import tqdm

        for prepared_id in tqdm(
            prepared_ids,
            desc="Refresh cache artifacts",
            unit="roi",
        ):
            self.db_client.rpc(
                "refresh_prepared_cache_artifacts",
                {"p_prepared_id": prepared_id},
            )
        logger.info(
            "[SupabaseStore] refreshed cache artifacts for %d prepared ROIs",
            len(prepared_ids),
        )


def create_store(mode: str, **kwargs) -> PipelineStore:
    if mode == "json":
        base_dir = kwargs.get("store_dir")
        if not base_dir:
            raise ValueError("store_dir is required for json store mode")
        return JsonFileStore(base_dir)
    elif mode == "db":
        db_client = kwargs.get("db_client")
        if not db_client:
            raise ValueError("db_client is required for db store mode")
        if getattr(db_client, "mode", None) == "local":
            return LocalPostgresStore(db_client)
        return SupabaseStore(db_client)
    else:
        raise ValueError(f"Unknown store mode: {mode!r}. Use 'json' or 'db'.")

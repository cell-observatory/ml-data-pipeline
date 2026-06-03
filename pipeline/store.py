from __future__ import annotations

import re
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


# ``pg_get_expr(relpartbound, ...)`` for RANGE partitions looks like
# ``FOR VALUES FROM (363) TO (375)`` (half-open interval on the key).
_PART_RANGE_BOUND_RE = re.compile(
    r"FOR VALUES FROM\s*\(\s*'?(-?\d+)'?\s*\)\s*TO\s*\(\s*'?(-?\d+)'?\s*\)",
    re.IGNORECASE,
)


def _relation_has_child_partitions(cur, regclass_fqn: str) -> bool:
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_inherits "
        "WHERE inhparent = %s::regclass)",
        (regclass_fqn,),
    )
    return bool(cur.fetchone()[0])


def _max_exclusive_upper_range_partition(
    cur, regclass_fqn: str,
) -> int | None:
    """Largest ``TO (hi)`` among RANGE children; ``None`` if no parseable bound."""
    cur.execute(
        """
        SELECT pg_get_expr(c.relpartbound, c.oid, true) AS bound
          FROM pg_catalog.pg_inherits i
          JOIN pg_catalog.pg_class c ON c.oid = i.inhrelid
         WHERE i.inhparent = %s::regclass
        """,
        (regclass_fqn,),
    )
    max_hi: int | None = None
    for (bound_expr,) in cur.fetchall():
        if not bound_expr:
            continue
        m = _PART_RANGE_BOUND_RE.search(bound_expr)
        if not m:
            continue
        hi = int(m.group(2))
        max_hi = hi if max_hi is None else max(max_hi, hi)
    return max_hi


def _ensure_range_partition_covers_int_key(
    cur,
    *,
    schema: str,
    table: str,
    key: int,
    span: int = 64,
) -> None:
    """Attach a RANGE partition on ``schema.table`` so ``key`` maps to a leaf.

    The production sandbox partitions ``prepared_cubes`` and
    ``annotations_3d`` by ``prepared_id`` / ``roi_id``.  New synthetic
    ROIs get SERIAL ids **past the last pre-made partition** (e.g. 385
    when the tarball only had partitions through ``TO (375``), and
    ``INSERT`` then errors with *no partition of relation found*.  Failed
    transactions still burn sequence values, which is why a run can show
    both that error **and** later failures on unrelated statements.

    If ``schema.table`` has no child partitions, this is a no-op (table
    not partitioned in this environment).
    """
    parent_fqn = f"{schema}.{table}"
    if not _relation_has_child_partitions(cur, parent_fqn):
        return

    max_end = _max_exclusive_upper_range_partition(cur, parent_fqn)
    if max_end is None:
        raise RuntimeError(
            f"{parent_fqn} has partitions but no RANGE (FROM..TO) bounds "
            f"this helper can parse — create a partition manually for key={key}"
        )

    # Half-open ranges: ``FROM (lo) TO (hi)`` accepts ``lo <= id < hi``.
    safety = 0
    while key >= max_end:
        safety += 1
        if safety > 512:
            raise RuntimeError(
                f"partition extension loop exceeded limit for {parent_fqn} "
                f"key={key}"
            )
        lo = max_end
        hi = max(lo + span, key + 1 + 8)
        hi_inclusive = hi - 1
        part_name = f"{table}_p{lo:05d}_p{hi_inclusive:05d}"
        ddl = sql.SQL(
            "CREATE TABLE {} PARTITION OF {} FOR VALUES FROM ({}) TO ({})"
        ).format(
            sql.Identifier(schema, part_name),
            sql.Identifier(schema, table),
            sql.Literal(lo),
            sql.Literal(hi),
        )
        cur.execute(ddl)
        logger.info(
            "[LocalPostgresStore] created missing partition %s on %s "
            "FOR VALUES FROM %s TO %s (key=%s needed coverage)",
            part_name,
            parent_fqn,
            lo,
            hi,
            key,
        )
        max_end = hi


# Hot tables to ANALYZE before per-ROI cache refresh.
_REFRESH_ANALYZE_TARGETS = (
    "public.prepared",
    "public.prepared_tiles",
    "public.prepared_tiles_view_table",
    "public.prepared_cubes",
)


def _refresh_cache_artifacts_via_pg(
    conn,
    prepared_ids: list[int],
    *,
    backend_label: str,
    statement_timeout: str = "2h",
    continue_on_error: bool = True,
) -> list[int]:
    """Run ANALYZE + per-ROI refresh_prepared_cache_artifacts(p_prepared_id).

    Used by both LocalPostgresStore and SupabaseStore (when the latter has a
    direct-Postgres write URI). Owns the connection — closes it in a finally.
    A long server-side ``statement_timeout`` is set so individual refresh
    calls don't get killed mid-flight.
    """
    from tqdm import tqdm

    failed_ids: list[int] = []

    try:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = '{statement_timeout}'")
            for target in tqdm(
                _REFRESH_ANALYZE_TARGETS,
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
                try:
                    cur.execute(f"SET statement_timeout = '{statement_timeout}'")
                    cur.execute(
                        "SELECT refresh_prepared_cache_artifacts(%s)",
                        [prepared_id],
                    )
                    conn.commit()
                except Exception:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    failed_ids.append(prepared_id)
                    logger.exception(
                        "[%s] cache refresh failed for prepared_id=%s",
                        backend_label,
                        prepared_id,
                    )
                    if not continue_on_error:
                        raise
        if failed_ids:
            logger.error(
                "[%s] cache refresh failed for %d prepared IDs: %s",
                backend_label,
                len(failed_ids),
                failed_ids,
            )
        logger.info(
            "[%s] cache refresh finished: %d succeeded, %d failed (direct PG)",
            backend_label,
            len(prepared_ids) - len(failed_ids),
            len(failed_ids),
        )
        return failed_ids
    finally:
        conn.close()


# Fixed column order for COPY into public.prepared_cubes. Must match what
# pipeline.ingest.build_prepared_cubes_entries produces (plus prepared_id,
# injected by the ingest step). Keep this in sync with the table schema.
_PREPARED_CUBES_COLUMNS: tuple[str, ...] = (
    "prepared_id",
    "tile_name",
    "chunk",
    "time",
    "z_start",
    "y_start",
    "x_start",
    "channel",
    "occupancy_ratio",
    "cdf_80",
    "cdf_90",
    "cdf_95",
    "cdf_99",
)


def _copy_prepared_cubes_via_pg(
    conn,
    prepared_id: int,
    cubes: list[dict],
) -> int:
    """Bulk-insert prepared_cubes via psycopg.
    Commits on success and rolls back on failure. Caller owns the connection
    lifecycle (open + close).
    """
    if not cubes:
        return 0

    col_idents = sql.SQL(", ").join(
        sql.Identifier(c) for c in _PREPARED_CUBES_COLUMNS
    )
    copy_stmt = sql.SQL("COPY {} ({}) FROM STDIN").format(
        sql.Identifier("public", "prepared_cubes"),
        col_idents,
    )

    n = 0
    try:
        with conn.cursor() as cur:
            _ensure_range_partition_covers_int_key(
                cur, schema="public", table="prepared_cubes", key=prepared_id,
            )
            with cur.copy(copy_stmt) as cp:
                for cube in cubes:
                    cp.write_row([
                        prepared_id if c == "prepared_id" else cube[c]
                        for c in _PREPARED_CUBES_COLUMNS
                    ])
                    n += 1
        conn.commit()
        return n
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise


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
        annotations: list[dict] | None = None,
    ) -> int:
        """Insert prepared + tiles + cubes (and optional annotations) atomically.

        Returns the DB-assigned prepared_id. ``annotations`` rows have
        their ``roi_id`` overwritten with that id, and any
        ``prepared_id`` field is dropped, so callers can pass the
        CSV-local id and let the store remap.

        Does NOT refresh aggregate caches. Callers should invoke
        ``refresh_cache_artifacts`` once after ingesting all ROIs.
        """
        ...

    @abstractmethod
    def mark_raw_roi_prepared(self, roi_acquisition_id: str, prepared_id: int) -> None:
        """Mark the source raw_rois row as prepared after ingest succeeds."""
        ...

    @abstractmethod
    def refresh_cache_artifacts(
        self,
        prepared_ids: list[int],
        *,
        statement_timeout: str = "2h",
        continue_on_error: bool = True,
    ) -> list[int]:
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
        annotations: list[dict] | None = None,
    ) -> int:
        roi_id = prepared.get("raw_roi_acquisition_id", "unknown")
        payload = {
            "prepared": prepared,
            "tiles": tiles,
            "cubes_count": len(cubes),
            "cubes_sample": cubes[:5],
            "annotations_count": len(annotations or []),
            "annotations_sample": (annotations or [])[:5],
        }
        path = self._prepared_dir / f"prepared_{roi_id}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        logger.info("[JsonFileStore] wrote prepared roi %s to %s", roi_id, path)
        return 0

    def mark_raw_roi_prepared(self, roi_acquisition_id: str, prepared_id: int) -> None:
        logger.info(
            "[JsonFileStore] would mark raw ROI %s as prepared_id=%s",
            roi_acquisition_id,
            prepared_id,
        )

    def refresh_cache_artifacts(
        self,
        prepared_ids: list[int],
        *,
        statement_timeout: str = "2h",
        continue_on_error: bool = True,
    ) -> list[int]:
        return []


class LocalPostgresStore(PipelineStore):
    """Writes directly to a local Postgres sandbox via SQL."""

    CUBE_INSERT_BATCH = 10_000
    ANNOTATION_INSERT_BATCH = 10_000
    _JSON_COLUMNS = {
        "roi_tile_scan_log": {"error_messages", "scan_metadata"},
        "prepared": {"channel_mapping"},
        "prepared_tiles": set(),
        "prepared_cubes": set(),
        "annotations_3d": set(),
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

    def _prepare_annotation_rows(
        self,
        prepared_id: int,
        annotations: list[dict],
        tile_names: set[str] | None = None,
    ) -> list[dict]:
        """Stamp roi_id, drop CSV-local join keys, and validate tile_id.

        ``tile_names``, when provided, is the authoritative set of
        tile_name values for ``prepared_id`` (typically built from the
        prepared_tiles dicts being inserted in the same transaction).
        Any annotation whose ``tile_id`` is not in this set is an
        orphan -- the platform's tile-annotation agg joins on
        ``a.tile_id = ptv.tile_name`` so orphans would silently never
        roll up. We refuse to insert them.
        """
        out: list[dict] = []
        bad_tiles: set = set()
        for ann in annotations:
            row = dict(ann)
            row.pop("prepared_id", None)
            row.pop("id", None)  # GENERATED ALWAYS identity — server assigns
            row["roi_id"] = prepared_id
            if tile_names is not None:
                tid = row.get("tile_id")
                if tid not in tile_names:
                    bad_tiles.add(tid)
            out.append(row)

        if bad_tiles:
            sample = sorted(bad_tiles, key=lambda x: ("" if x is None else str(x)))[:5]
            raise ValueError(
                f"prepared_id={prepared_id}: {len(bad_tiles)} annotation "
                f"tile_id values are not among the {len(tile_names or ())} "
                f"prepared_tiles being inserted (e.g. {sample}). "
                f"Refusing to insert orphaned annotations -- they would "
                f"never join in aggregate_prepared_tile_annotation_agg_1."
            )
        return out

    def _insert_annotations_with_cur(
        self,
        cur,
        prepared_id: int,
        prepared_rows: list[dict],
    ) -> None:
        """Batched INSERT using an existing cursor (no commit/rollback)."""
        for i in range(0, len(prepared_rows), self.ANNOTATION_INSERT_BATCH):
            batch = prepared_rows[i : i + self.ANNOTATION_INSERT_BATCH]
            self._insert_many(cur, "annotations_3d", batch)

        # Inside the same transaction, confirm the rows actually landed
        # under this roi_id with is_consensus=true (the platform agg
        # filter). Mismatch is almost always a sign the caller mixed
        # roi_ids in `prepared_rows` -- abort the whole tx instead of
        # leaving a half-ingested ROI.
        cur.execute(
            "SELECT count(*) FROM public.annotations_3d "
            "WHERE roi_id = %s AND is_consensus = true",
            (prepared_id,),
        )
        result = cur.fetchone()
        observed = int(result[0]) if result else 0
        expected = sum(1 for r in prepared_rows if r.get("is_consensus"))
        if observed != expected:
            raise RuntimeError(
                f"prepared_id={prepared_id}: post-INSERT verification "
                f"failed: shipped {expected} consensus annotation rows "
                f"but DB reports {observed} for roi_id={prepared_id}. "
                f"Aborting transaction."
            )

    def ingest_prepared_roi(
        self,
        prepared: dict,
        tiles: list[dict],
        cubes: list[dict],
        annotations: list[dict] | None = None,
    ) -> int:
        annotations = annotations or []

        conn = self._connect()
        try:
            with conn.cursor() as cur:
                prepared_id = self._insert_row(cur, "prepared", prepared, returning="id")

                _ensure_range_partition_covers_int_key(
                    cur, schema="public", table="prepared_cubes", key=prepared_id,
                )

                for tile in tiles:
                    tile["prepared_id"] = prepared_id
                self._insert_many(cur, "prepared_tiles", tiles)

                for cube in cubes:
                    cube["prepared_id"] = prepared_id
                for i in range(0, len(cubes), self.CUBE_INSERT_BATCH):
                    batch = cubes[i : i + self.CUBE_INSERT_BATCH]
                    self._insert_many(cur, "prepared_cubes", batch)

                n_anno = 0
                if annotations:
                    _ensure_range_partition_covers_int_key(
                        cur, schema="public", table="annotations_3d", key=prepared_id,
                    )
                    tile_names = {t.get("tile_name") for t in tiles}
                    prepared_anno = self._prepare_annotation_rows(
                        prepared_id, annotations, tile_names=tile_names,
                    )
                    self._insert_annotations_with_cur(cur, prepared_id, prepared_anno)
                    n_anno = len(prepared_anno)

            conn.commit()
            logger.debug(
                "[LocalPostgresStore] ingested prepared_id=%d "
                "(%d tiles, %d cubes, %d annotations)",
                prepared_id,
                len(tiles),
                len(cubes),
                n_anno,
            )
            return prepared_id
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ingest_annotations_3d(
        self,
        prepared_id: int,
        annotations: list[dict],
    ) -> int:
        """Insert annotation rows into public.annotations_3d (standalone).

        Prefer passing ``annotations`` to ``ingest_prepared_roi`` so the
        whole ROI lands in one transaction. This entry point is for the
        case where you're back-filling annotations onto an already-
        ingested prepared row; it cannot validate ``tile_id`` against
        prepared_tiles for you, so the caller must guarantee that
        ``tile_id`` matches an existing ``prepared_tiles.tile_name`` row
        for ``prepared_id``.

        Each row's ``roi_id`` is overwritten with the supplied
        ``prepared_id``, and any ``prepared_id`` field is dropped.
        Returns the number of rows inserted.
        """
        if not annotations:
            return 0
        prepared_rows = self._prepare_annotation_rows(
            prepared_id, annotations, tile_names=None,
        )
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                _ensure_range_partition_covers_int_key(
                    cur, schema="public", table="annotations_3d", key=prepared_id,
                )
                self._insert_annotations_with_cur(cur, prepared_id, prepared_rows)
            conn.commit()
            logger.debug(
                "[LocalPostgresStore] inserted %d annotations for prepared_id=%d",
                len(prepared_rows),
                prepared_id,
            )
            return len(prepared_rows)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def mark_raw_roi_prepared(self, roi_acquisition_id: str, prepared_id: int) -> None:
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE public.raw_rois
                       SET is_prepared = true,
                           prepared_id = %s
                     WHERE roi_acquisition_id = %s
                    """,
                    (prepared_id, roi_acquisition_id),
                )
                if cur.rowcount != 1:
                    raise ValueError(
                        "Expected to update 1 raw_rois row for "
                        f"roi_acquisition_id={roi_acquisition_id!r}, "
                        f"updated {cur.rowcount}"
                    )
            conn.commit()
            logger.debug(
                "[LocalPostgresStore] marked raw ROI %s as prepared_id=%d",
                roi_acquisition_id,
                prepared_id,
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def refresh_cache_artifacts(
        self,
        prepared_ids: list[int],
        *,
        statement_timeout: str = "2h",
        continue_on_error: bool = True,
    ) -> list[int]:
        if not prepared_ids:
            return []
        return _refresh_cache_artifacts_via_pg(
            self._connect(),
            prepared_ids,
            backend_label="LocalPostgresStore",
            statement_timeout=statement_timeout,
            continue_on_error=continue_on_error,
        )


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
        annotations: list[dict] | None = None,
    ) -> int:
        # SupabaseStore can't ingest annotations through the REST API right now,
        # so refuse loudly rather than silently dropping rows. Real (non-synth)
        # data sets never pass this kwarg.
        if annotations:
            raise NotImplementedError(
                f"SupabaseStore.ingest_prepared_roi does not support inserting "
                f"annotations (received {len(annotations)} rows). Use the "
                f"local sandbox path (LocalPostgresStore) for synthetic data, "
                f"or extend SupabaseStore to POST into annotations_3d via REST."
            )
        # Cubes are the bulk of write traffic (~O(100k) rows per ROI). When a
        # direct-Postgres write URI is configured (SUPABASE_<MODE>_PG_WRITE_URI)
        # we COPY them straight to the database, skipping PostgREST and the
        # Supabase REST gateway entirely. ``prepared`` and ``prepared_tiles``
        # stay on REST: they're small and we need RETURNING id from prepared.
        prepared_id = None
        pg_conn = None
        try:
            try:
                pg_conn = self.db_client.pg_write_connection()
            except ValueError:
                pg_conn = None
            except Exception as exc:
                logger.warning(
                    "[SupabaseStore] direct-PG cube path unavailable (%s); "
                    "falling back to REST for cubes",
                    exc,
                )
                pg_conn = None

            resp = self.client.table("prepared").insert(prepared).execute()
            prepared_id = resp.data[0]["id"]

            for tile in tiles:
                tile["prepared_id"] = prepared_id
            self.client.table("prepared_tiles").insert(tiles).execute()

            if pg_conn is not None:
                _copy_prepared_cubes_via_pg(pg_conn, prepared_id, cubes)
                cube_path = "PG-COPY"
            else:
                for cube in cubes:
                    cube["prepared_id"] = prepared_id
                for i in range(0, len(cubes), self.CUBE_INSERT_BATCH):
                    batch = cubes[i : i + self.CUBE_INSERT_BATCH]
                    self.client.table("prepared_cubes").insert(batch).execute()
                cube_path = "REST"

            logger.debug(
                "[SupabaseStore] ingested prepared_id=%d (%d tiles, %d cubes via %s)",
                prepared_id,
                len(tiles),
                len(cubes),
                cube_path,
            )
            return prepared_id

        except Exception:
            if prepared_id is not None:
                logger.warning("[SupabaseStore] rolling back prepared_id=%d", prepared_id)
                self.client.table("prepared").delete().eq("id", prepared_id).execute()
            raise
        finally:
            if pg_conn is not None:
                try:
                    pg_conn.close()
                except Exception:
                    pass

    def mark_raw_roi_prepared(self, roi_acquisition_id: str, prepared_id: int) -> None:
        resp = (
            self.client.table("raw_rois")
            .update({"is_prepared": True, "prepared_id": prepared_id})
            .eq("roi_acquisition_id", roi_acquisition_id)
            .execute()
        )
        updated = len(resp.data or [])
        if updated != 1:
            raise ValueError(
                "Expected to update 1 raw_rois row for "
                f"roi_acquisition_id={roi_acquisition_id!r}, updated {updated}"
            )
        logger.debug(
            "[SupabaseStore] marked raw ROI %s as prepared_id=%d",
            roi_acquisition_id,
            prepared_id,
        )

    def refresh_cache_artifacts(
        self,
        prepared_ids: list[int],
        *,
        statement_timeout: str = "2h",
        continue_on_error: bool = True,
    ) -> list[int]:
        if not prepared_ids:
            return []

        # Prefer direct Postgres if SUPABASE_<MODE>_PG_WRITE_URI is set:
        # refresh_prepared_cache_artifacts can take many minutes per ROI.
        try:
            conn = self.db_client.pg_write_connection()
        except ValueError:
            from tqdm import tqdm

            logger.warning(
                "[SupabaseStore] no PG_WRITE_URI configured for mode=%s; "
                "falling back to RPC over REST (subject to gateway timeout)",
                self.db_client.mode,
            )
            failed_ids: list[int] = []
            for prepared_id in tqdm(
                prepared_ids,
                desc="Refresh cache artifacts (REST)",
                unit="roi",
            ):
                try:
                    self.db_client.rpc(
                        "refresh_prepared_cache_artifacts",
                        {"p_prepared_id": prepared_id},
                    )
                except Exception:
                    failed_ids.append(prepared_id)
                    logger.exception(
                        "[SupabaseStore] REST cache refresh failed for prepared_id=%s",
                        prepared_id,
                    )
                    if not continue_on_error:
                        raise
            if failed_ids:
                logger.error(
                    "[SupabaseStore] REST cache refresh failed for %d prepared IDs: %s",
                    len(failed_ids),
                    failed_ids,
                )
            logger.info(
                "[SupabaseStore] cache refresh finished: %d succeeded, %d failed (REST)",
                len(prepared_ids) - len(failed_ids),
                len(failed_ids),
            )
            return failed_ids

        return _refresh_cache_artifacts_via_pg(
            conn,
            prepared_ids,
            backend_label="SupabaseStore",
            statement_timeout=statement_timeout,
            continue_on_error=continue_on_error,
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

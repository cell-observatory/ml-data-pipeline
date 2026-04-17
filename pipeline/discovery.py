from __future__ import annotations

import logging
from typing import Optional

import pyarrow as pa

logger = logging.getLogger(__name__)


def discover_unprocessed_rois(
    db_client,
) -> pa.Table:
    """Query raw_rois for ROIs that have not yet been prepared.

    Returns a PyArrow table with columns:
        roi_acquisition_id, roi_path, qc, channel_pattern_metadata,
        biological_channel_to_tiff_tokens
    """
    sql = """
        SELECT
            roi_acquisition_id,
            roi_path,
            qc,
            channel_pattern_metadata,
            biological_channel_to_tiff_tokens
        FROM public.raw_rois
        WHERE is_prepared = false
    """

    sql += " ORDER BY roi_acquisition_id"

    table = db_client.query_arrow(sql)
    return table

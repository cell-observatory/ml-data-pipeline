from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def resolve_channel_mapping(
    requested_channels: list[str],
    biological_channel_to_tiff_tokens: dict[str, str],
    channel_aliases: dict[str, list[str]],
) -> Optional[dict[str, str]]:
    """Match requested channel names against the ROI's bio channel keys.

    Each requested name (e.g. ``"membrane"``) is expanded via
    *channel_aliases* to all its synonyms (e.g. ``["membrane"]``), then
    each synonym is checked as a **substring** of the bio-channel keys
    (e.g. ``"tdmstaygold-membrane"``).

    Returns a dict mapping channel index (str) to the **full** bio-channel
    name so that downstream lookups (scanner, tiff-token resolution) work
    with exact keys.  Returns ``None`` if any requested channel cannot be
    uniquely matched.
    """
    available = list(biological_channel_to_tiff_tokens.keys())
    mapping: dict[str, str] = {}
    used: set[str] = set()

    for i, requested in enumerate(requested_channels):
        synonyms = channel_aliases.get(requested, [requested])
        matches = [
            ch for ch in available
            if any(syn in ch for syn in synonyms) and ch not in used
        ]
        if len(matches) == 0:
            return None
        mapping[str(i)] = matches[0]
        used.add(matches[0])

    return mapping


def resolve_tiff_tokens_for_mapping(
    channel_mapping: dict[str, str],
    biological_channel_to_tiff_tokens: dict[str, str],
) -> dict[str, str]:
    """Given a resolved channel mapping, return channel_index -> tiff_token."""
    return {
        ch_idx: biological_channel_to_tiff_tokens[bio_name]
        for ch_idx, bio_name in channel_mapping.items()
    }

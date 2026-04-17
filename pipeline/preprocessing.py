from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROFILES_FILENAME = "preprocessing_profiles.json"


@dataclass
class PreprocessingProfile:
    name: str
    match_channels: list[str]
    match_roi_path_contains: Optional[str] = None

    csc_unmixing: Optional[dict[str, Any]] = None
    flatfield_paths: Optional[list[str]] = None
    background_folder: Optional[str] = None

    def to_dataset_fields(self) -> dict[str, Any]:
        """Return the fields that should be merged into a dataset/ROIPlan preprocessing dict."""
        out: dict[str, Any] = {}
        if self.csc_unmixing is not None:
            out["csc_unmixing"] = True
            out["chromatic_offset"] = self.csc_unmixing["chromatic_offset"]
            out["unmix_pairs"] = self.csc_unmixing["unmix_pairs"]
        else:
            out["csc_unmixing"] = False
        if self.flatfield_paths is not None:
            out["flatfield_paths"] = self.flatfield_paths
        if self.background_folder is not None:
            out["background_folder"] = self.background_folder
        return out


def load_profiles(
    repo_root: str | Path | None = None,
) -> tuple[list[PreprocessingProfile], dict[str, list[str]]]:
    """Load preprocessing profiles and channel aliases from the JSON file.

    Returns ``(profiles, channel_aliases)``.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    profiles_path = Path(repo_root) / PROFILES_FILENAME
    if not profiles_path.exists():
        raise FileNotFoundError(
            f"Preprocessing profiles not found at {profiles_path}. "
            f"Create {PROFILES_FILENAME} in the repo root."
        )

    with open(profiles_path, "r") as f:
        data = json.load(f)

    channel_aliases: dict[str, list[str]] = data.get("channel_aliases", {})

    profiles = []
    for entry in data.get("profiles", []):
        profiles.append(
            PreprocessingProfile(
                name=entry["name"],
                match_channels=sorted(entry["match_channels"]),
                match_roi_path_contains=entry.get("match_roi_path_contains"),
                csc_unmixing=entry.get("csc_unmixing"),
                flatfield_paths=entry.get("flatfield_paths"),
                background_folder=entry.get("background_folder"),
            )
        )

    logger.info("Loaded %d preprocessing profiles from %s", len(profiles), profiles_path)
    return profiles, channel_aliases


def _canonicalize(channel: str, channel_aliases: dict[str, list[str]]) -> str:
    """Map a synonym to its canonical name.  Returns as-is if not found."""
    for canonical, synonyms in channel_aliases.items():
        if channel in synonyms:
            return canonical
    return channel


def match_profile(
    requested_channels: list[str],
    roi_path: str,
    profiles: list[PreprocessingProfile],
    channel_aliases: dict[str, list[str]],
) -> PreprocessingProfile | None:
    """Find the first profile matching the user's requested channels.

    *requested_channels* are the names the user passed on the CLI (e.g.
    ``["membrane", "histone"]``).  They are canonicalized via
    *channel_aliases* before comparison with each profile's
    ``match_channels`` (which are already canonical).
    """
    canonical_requested = sorted(
        _canonicalize(ch, channel_aliases) for ch in requested_channels
    )

    for profile in profiles:
        if profile.match_channels != canonical_requested:
            continue

        if profile.match_roi_path_contains is not None:
            if profile.match_roi_path_contains not in roi_path:
                continue

        logger.debug(
            "Matched profile %r for channels=%s roi_path=%s",
            profile.name, canonical_requested, roi_path,
        )
        return profile

    return None

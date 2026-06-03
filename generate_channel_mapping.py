"""Generate channel_mapping.json from a synthetic *_cytosol_fp_expressions.csv.

Output format::

    {"0": "<membrane-fp>-membrane",
     "1": "<fp1>-cytosol",
     "2": "<fp2>-cytosol",
     ...,
     "<N>": "instance-mask"}

The membrane FP name is provided as a flag (the membrane CSV has no FP
column). Cytosol FP names are taken from the column headers of the
``*_cytosol_fp_expressions.csv`` (excluding ``label_id`` and ``timepoint``)
in their original order, so channel indices stay stable across reruns.

The final entry is the dense instance-id labelmap that the
cell_observatory_platform dataloader strips off as ``inputs[..., -1]``.
It is required to be the highest index so that ``channel_size`` in the DB
matches ``zarr.shape[-1]``; see the ``MASK_ROLE`` invariant in
``build_synthetic_prepared_csvs.py``.

The naming convention ``<fp>-<role>`` is preserved for the mask too
(``instance-mask``) so:

  - ``build_synthetic_prepared_csvs.py``'s parser doesn't need a special
    case for mask entries
  - ``cell_observatory_platform.data.datasets.utils.resolve_channel_localization_indices``
    can resolve ``"mask"`` (or ``"labelmap"``) to the correct index if a
    config ever asks for it explicitly

This is a temporary helper. Future synthetic-data generation iterations
will emit ``channel_mapping.json`` themselves and this script will be
retired.
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd


NON_FP_COLUMNS = {"label_id", "timepoint"}

# Convention: see build_synthetic_prepared_csvs.MASK_ROLE / INTENSITY_ROLES.
# Keep these strings in sync there.
MASK_FP_NAME = "instance"
MASK_ROLE = "mask"


def build_mapping(membrane_fp_name: str,
                  cytosol_fp_names: list[str]) -> dict[str, str]:
    """Assemble the channel index -> '<fp>-<role>' string mapping.

    The mask channel is always the highest-indexed entry.
    """
    mapping: dict[str, str] = {"0": f"{membrane_fp_name}-membrane"}
    for i, fp in enumerate(cytosol_fp_names, start=1):
        mapping[str(i)] = f"{fp}-cytosol"
    mapping[str(len(mapping))] = f"{MASK_FP_NAME}-{MASK_ROLE}"
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--membrane-fp-name", required=True,
        help="Name of the fluorescent protein rendered into the membrane "
             "channel (e.g. tdmstaygold).",
    )
    ap.add_argument(
        "--fp-expressions-csv", required=True,
        help="Path to a *_cytosol_fp_expressions.csv. Cytosol channel "
             "names are taken from its non-label columns in order.",
    )
    ap.add_argument(
        "--output", required=True,
        help="Path to write channel_mapping.json",
    )
    args = ap.parse_args()

    cols = pd.read_csv(args.fp_expressions_csv, nrows=0).columns.tolist()
    cytosol_fps = [c for c in cols if c not in NON_FP_COLUMNS]
    if not cytosol_fps:
        raise ValueError(
            f"{args.fp_expressions_csv}: no FP columns found (columns={cols})"
        )

    mapping = build_mapping(args.membrane_fp_name, cytosol_fps)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(mapping)} channels to {args.output}: {mapping}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
import re
import json
import logging
import inspect
from datetime import datetime

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pipeline.store import PipelineStore
from pipeline.scanner import ROIScanResult
from pipeline.planner import ProcessingPlan
from pipeline.ingest import build_prepared_entry, build_prepared_tiles_entries, build_prepared_cubes_entries

logger = logging.getLogger(__name__)


def _load_valid_params() -> dict[str, bool]:
    """Load PetaKit5D valid parameter names by inspecting wrapper source."""
    try:
        from PyPetaKit5D import XR_chromatic_shift_correction_data_wrapper
        from PyPetaKit5D import XR_unmix_channels_data_wrapper
        from PyPetaKit5D import XR_decon_data_wrapper
        from PyPetaKit5D import XR_deskew_rotate_data_wrapper

        text = ""
        for wrapper in [
            XR_chromatic_shift_correction_data_wrapper,
            XR_unmix_channels_data_wrapper,
            XR_decon_data_wrapper,
            XR_deskew_rotate_data_wrapper,
        ]:
            with open(inspect.getfile(wrapper), "r", encoding="utf-8") as f:
                text += f.read()

        matches = re.findall(r'"(.*?)": \[', text)
        return {key: True for key in matches}
    except ImportError:
        logger.warning("PyPetaKit5D not available, skipping valid_params check")
        return {}


def execute_plan(
    plan: ProcessingPlan,
    scan_results: list[ROIScanResult],
    store: PipelineStore,
    log_dir: str,
    background_folder: str = "",
    add_support_ratio_metadata: bool = False,
    skip_processing: bool = False,
    output_date_ymd: list[str] | None = None,
    cluster: str | None = None,
    refresh_statement_timeout: str = "2h",
    refresh_continue_on_error: bool = True,
) -> list[int]:
    """Execute a ProcessingPlan: run Slurm jobs and ingest results.

    When *skip_processing* is True the heavy CSC/decon/DSR/conversion jobs are
    skipped entirely.  The metadata skeleton is still rebuilt from the raw scan
    and then validated against the existing ``metadata.json`` on disk (which must
    be present).  *output_date_ymd* is required in this mode so the pipeline can
    locate the already-processed output folder.

    Returns a list of prepared_ids (one per ROI).
    """
    datasets = plan.to_datasets_dict()

    if skip_processing and output_date_ymd is None:
        raise ValueError(
            "skip_processing=True requires output_date_ymd so the pipeline can "
            "locate the already-processed output folder."
        )

    date_ymd = output_date_ymd or [
        str(datetime.now().year),
        str(datetime.now().month),
        str(datetime.now().day),
    ]

    if skip_processing:
        from submit_jobs import (
            build_metadata_from_existing,
            merge_existing_metadata,
            OCCUPANCY_CHUNK_FIELDS,
            SUPPORT_RATIO_CHUNK_FIELDS,
        )

        datasets, _folders_to_delete = build_metadata_from_existing(
            datasets, plan.output_folder, date_ymd,
        )

        required_chunk_fields = list(OCCUPANCY_CHUNK_FIELDS)
        if add_support_ratio_metadata:
            required_chunk_fields.extend(SUPPORT_RATIO_CHUNK_FIELDS)

        datasets = merge_existing_metadata(
            datasets,
            required_chunk_fields=required_chunk_fields,
        )

        # Split the full output_folder into server_folder + relative path
        # (normally done inside write_metadata_files) without writing to disk.
        for _folder_path, dataset in datasets.items():
            out_folder = dataset.get("metadata", {}).get("output_folder", "")
            match = re.search(r"(/\d+/\d{1,2}/\d{1,2}/)", out_folder)
            if match:
                split_index = match.start(1)
                dataset["metadata"]["server_folder"] = out_folder[:split_index]
                dataset["metadata"]["output_folder"] = out_folder[split_index:].lstrip("/")
    else:
        from submit_jobs import (
            submit_csc_unmixing_jobs,
            submit_decon_dsr_jobs,
            submit_training_image_jobs,
            collect_occupancy_metadata,
            collect_support_ratio_metadata,
            write_metadata_files,
        )

        input_file_path = os.path.join(log_dir, f"datasets_{plan.run_id}.json")
        with open(input_file_path, "w") as f:
            json.dump(datasets, f, indent=2)

        data_shape = [plan.batch_size, *plan.cube_shape]
        inner_chunk_shape = [1, 32, 32, 32]
        valid_params = _load_valid_params()

        submit_csc_unmixing_jobs(
            datasets, plan.batch_size, input_file_path, background_folder, log_dir, valid_params
        )

        decon_dsr_job_times = submit_decon_dsr_jobs(
            datasets, plan.batch_size, valid_params, input_file_path, log_dir
        )

        datasets, _folders_to_delete = submit_training_image_jobs(
            datasets, plan.output_folder, data_shape, inner_chunk_shape, plan.batch_size,
            date_ymd, plan.output_zarr_version, input_file_path, log_dir, decon_dsr_job_times,
        )

        datasets = collect_occupancy_metadata(datasets)

        if add_support_ratio_metadata:
            datasets = collect_support_ratio_metadata(
                datasets, 2, plan.output_zarr_version, log_dir
            )

        datasets = write_metadata_files(datasets)

    prepared_ids: list[int] = []
    with logging_redirect_tqdm():
        pbar = tqdm(
            zip(plan.rois, scan_results),
            total=len(plan.rois),
            desc="Ingest prepared ROIs",
            unit="roi",
        )
        for roi_plan, scan in pbar:
            metadata = datasets.get(roi_plan.roi_path, {}).get("metadata", {})

            prepared = build_prepared_entry(
                scan,
                roi_plan.channel_mapping,
                metadata=metadata,
                cube_size=plan.cube_shape[0],
                channel_size=roi_plan.channel_size,
                cluster=cluster,
            )

            tiles = build_prepared_tiles_entries(
                scan,
                metadata=metadata,
                channel_size=roi_plan.channel_size,
            )

            cubes = build_prepared_cubes_entries(
                metadata,
                output_zarr_version=plan.output_zarr_version,
            )

            prepared_id = store.ingest_prepared_roi(prepared, tiles, cubes)
            store.mark_raw_roi_prepared(scan.roi_acquisition_id, prepared_id)
            prepared_ids.append(prepared_id)
            pbar.set_postfix(
                prepared_id=prepared_id,
                n_tiles=len(tiles),
                n_cubes=len(cubes),
            )

        refresh_failures = store.refresh_cache_artifacts(
            prepared_ids,
            statement_timeout=refresh_statement_timeout,
            continue_on_error=refresh_continue_on_error,
        )
        if refresh_failures:
            failure_dir = log_dir
            os.makedirs(failure_dir, exist_ok=True)
            failure_path = os.path.join(
                failure_dir,
                f"refresh_failures_{plan.run_id}.txt",
            )
            with open(failure_path, "w") as f:
                f.write("\n".join(str(pid) for pid in refresh_failures))
                f.write("\n")
            logger.error(
                "Cache refresh failed for %d prepared IDs: %s. "
                "Wrote retry list to %s",
                len(refresh_failures),
                refresh_failures,
                failure_path,
            )

    return prepared_ids

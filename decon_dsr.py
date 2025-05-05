import argparse
import inspect
import json
import os
import re

from PyPetaKit5D import XR_decon_data_wrapper
from PyPetaKit5D import XR_deskew_rotate_data_wrapper


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Path to the json file containing dataset information")
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument("--decon", action="store_true", help="Run decon")
    ap.add_argument("--dsr", action="store_true", help="Run dsr")
    ap.add_argument('--background-paths', type=lambda s: list(map(str, s.split(','))), default=None,
                    help="Paths to the background files separated by comma")
    ap.add_argument('--flatfield-paths', type=lambda s: list(map(str, s.split(','))), default=None,
                    help="Paths to the  separated by comma")
    args = ap.parse_args()
    input_file = args.input_file
    folder_paths = args.folder_paths
    channel_patterns = args.channel_patterns
    background_paths = args.background_paths
    flatfield_paths = args.flatfield_paths
    decon = args.decon
    dsr = args.dsr

    with open(input_file) as f:
        datasets_json = json.load(f)

    datasets = {}
    for folder_path in folder_paths:
        datasets[folder_path] = datasets_json[folder_path]

    with open(inspect.getfile(XR_decon_data_wrapper), "r", encoding="utf-8") as f:
        decon_text = f.read()

    with open(inspect.getfile(XR_deskew_rotate_data_wrapper), "r", encoding="utf-8") as f:
        dsr_text = f.read()

    matches = re.findall(r'"(.*?)": \[', decon_text)
    valid_decon_params = {key: True for key in matches}

    matches = re.findall(r'"(.*?)": \[', dsr_text)
    valid_dsr_params = {key: True for key in matches}

    run_decon = getattr(__import__('PyPetaKit5D'), 'XR_decon_data_wrapper')
    run_dsr = getattr(__import__("PyPetaKit5D"), 'XR_deskew_rotate_data_wrapper')

    if decon:
        for folder_path, dataset in datasets.items():
            kwargs = {}
            for param, value in dataset.items():
                if param in  valid_decon_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            kwargs['saveZarr'] = True
            run_decon([folder_path], **kwargs)
    if dsr:
        for folder_path, dataset in datasets.items():
            kwargs = {}
            for param, value in dataset.items():
                if param in valid_dsr_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            kwargs['saveZarr'] = True
            if background_paths:
                kwargs['FFImagePaths'] = flatfield_paths
                kwargs['backgroundPaths'] = background_paths
            if decon:
                kwargs['zarrFile'] = True
                if 'resultDirName' in dataset:
                    folder_path = os.path.join(folder_path, dataset['resultDirName'])
                else:
                    folder_path = os.path.join(folder_path, 'matlab_decon')
            run_dsr([folder_path], **kwargs)

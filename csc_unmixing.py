import argparse
import inspect
import json
import os
import re

from PyPetaKit5D import XR_chromatic_shift_correction_data_wrapper
from PyPetaKit5D import XR_unmix_channels_data_wrapper

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Path to the json file containing dataset information")
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument("--csc-unmixing", action="store_true", help="Run csc and unmixing")
    ap.add_argument('--background-paths', type=lambda s: list(map(str, s.split(','))), default=None,
                    help="Paths to the background files separated by comma")
    ap.add_argument('--flatfield-paths', type=lambda s: list(map(str, s.split(','))), default=None,
                    help="Paths to the  separated by comma")
    args = ap.parse_args()
    input_file = args.input_file
    folder_paths = args.folder_paths
    channel_patterns = args.channel_patterns
    csc_unmixing = args.csc_unmixing
    background_paths = args.background_paths
    flatfield_paths = args.flatfield_paths

    with open(input_file) as f:
        datasets_json = json.load(f)

    datasets = {}
    for folder_path in folder_paths:
        datasets[folder_path] = datasets_json[folder_path]

    with open(inspect.getfile(XR_chromatic_shift_correction_data_wrapper), "r", encoding="utf-8") as f:
        csc_text = f.read()

    with open(inspect.getfile(XR_unmix_channels_data_wrapper), "r", encoding="utf-8") as f:
        unmixing_text = f.read()

    matches = re.findall(r'"(.*?)": \[', csc_text)
    valid_csc_params = {key: True for key in matches}

    matches = re.findall(r'"(.*?)": \[', unmixing_text)
    valid_unmixing_params = {key: True for key in matches}

    run_csc = getattr(__import__('PyPetaKit5D'), 'XR_chromatic_shift_correction_data_wrapper')
    run_unmixing = getattr(__import__('PyPetaKit5D'), 'XR_unmix_channels_data_wrapper')

    if csc_unmixing:
        for folder_path, dataset in datasets.items():
            kwargs = {}
            for param, value in dataset.items():
                if param in valid_csc_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            kwargs['saveZarr'] = True
            kwargs['chromaticOffset'] = [[0, 0, 0],[5, -12, -1]]
            run_csc([folder_path], **kwargs)
            kwargs = {}
            for param, value in dataset.items():
                if param in valid_unmixing_params:
                    kwargs[param] = value
            kwargs['channelPatterns'] = channel_patterns
            kwargs['zarrFile'] = True
            kwargs['saveZarr'] = True
            kwargs['unmixFactors'] = [-0.112, 1]
            kwargs['channelInd'] = 2
            if background_paths:
                kwargs['FFImagePaths'] = flatfield_paths
                kwargs['backgroundPaths'] = background_paths
            if 'resultDirName' in dataset:
                folder_path = os.path.join(folder_path, dataset['resultDirName'])
            else:
                folder_path = os.path.join(folder_path, 'Chromatic_Shift_Corrected')
            run_unmixing([folder_path], **kwargs)
            kwargs['channelPatterns'] = [channel_patterns[0]]
            kwargs['unmixFactors'] = [1]
            kwargs['channelInd'] = 1
            if background_paths:
                kwargs['FFImagePaths'] = [flatfield_paths[0]]
                kwargs['backgroundPaths'] = [background_paths[0]]
            run_unmixing([folder_path], **kwargs)

import argparse
import copy
import glob
import subprocess
import math
import json
import os
import sys
import shutil
from importlib.metadata import version
from pathlib import Path

import tensorstore as ts

from convert_files import get_chunk_bboxes, get_filenames
import inspect
import re
from datetime import datetime
import time
from PyPetaKit5D import XR_decon_data_wrapper
from PyPetaKit5D import XR_deskew_rotate_data_wrapper


def create_sbatch_matlab_script(fn, fn_psf, chunk_i, timepoint_i, channel_i, log_dir=''):
    cpus_per_task = 1
    return f'''#!/bin/sh
#SBATCH --qos=abc_high
#SBATCH --partition=abc
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu=21000
#SBATCH --output={log_dir}/%j.out
cd {os.path.dirname(os.path.abspath(__file__))};matlab -batch \
"python_FFT2OTF_support_ratio(\'{fn}\',\'{fn_psf}\',{chunk_i},{timepoint_i},{channel_i},\'{sys.executable}\')"
'''


def create_sbatch_script(python_script_name, input_file, folder_path, channel_pattern, tiled=True, channel_num=0, output_folder='',
                         output_name_start_num=0, batch_start_number=0,
                         batch_size=16, input_is_zarr=False, date_ymd=None, elapsed_sec=0, orig_folder_path=None,
                         log_dir='', decon=False, dsr=False):
    extra_params = ''
    cpus_per_task = 24
    if python_script_name == 'convert_files.py':
        extra_params = f'''--channel-num {channel_num} --input-file {input_file} --output-folder {output_folder} --output-name-start-num {output_name_start_num} --batch-start-number {batch_start_number} --batch-size {batch_size} --elapsed-sec {elapsed_sec} '''
        if tiled:
            extra_params += '--tiled '
        if input_is_zarr:
            extra_params += '--input-is-zarr '
        if date_ymd:
            extra_params += f'--date {",".join(map(str, date_ymd))} '
        if orig_folder_path:
            extra_params += f'--orig-folder-paths {orig_folder_path} '
    elif python_script_name == 'decon_dsr.py':
        cpus_per_task = 24
        if input_file:
            extra_params += f'--input-file {input_file} '
        if decon:
            extra_params += '--decon '
        if dsr:
            extra_params += '--dsr '
    return f'''#!/bin/sh
#SBATCH --qos=abc_high
#SBATCH --partition=abc
#SBATCH --account=co_abc
#SBATCH --nodes=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem-per-cpu=21000
#SBATCH --output={log_dir}/%j.out
{sys.executable} \
{os.path.dirname(os.path.abspath(__file__))}/{python_script_name} \
--folder-paths {folder_path} --channel-patterns {channel_pattern} {extra_params}
'''


def create_file_conversion_sbatch_script(input_file, folder_path, channel_pattern, tiled, channel_num, output_folder, output_name_start_num,
                                         batch_start_number, batch_size,
                                         input_is_zarr, date_ymd, elapsed_sec, orig_folder_path, log_dir):
    return create_sbatch_script('convert_files.py', input_file, folder_path, channel_pattern, tiled=tiled,
                                channel_num=channel_num, output_folder=output_folder,
                                output_name_start_num=output_name_start_num, batch_start_number=batch_start_number,
                                batch_size=batch_size,
                                input_is_zarr=input_is_zarr, date_ymd=date_ymd, elapsed_sec=elapsed_sec,
                                orig_folder_path=orig_folder_path, log_dir=log_dir)


def create_decon_dsr_sbatch_script(input_file, folder_path, channel_pattern, decon, dsr, log_dir):
    if decon is None:
        decon = False
    if dsr is None:
        dsr = False
    return create_sbatch_script('decon_dsr.py', input_file, folder_path, channel_pattern, decon=decon, dsr=dsr,
                                log_dir=log_dir)


class Dataset:
    def __init__(self, channel_patterns, decon, dsr):
        self.channel_patterns = channel_patterns
        if decon is not None:
            self.decon = json.loads(decon.lower())
        else:
            self.decon = False
        if dsr is not None:
            self.dsr = json.loads(dsr.lower())
        else:
            self.dsr = False
        self.input_is_zarr = False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Path to the json file containing dataset information")
    ap.add_argument('--output-folder', type=str, required=True,
                    help="Path to the folder to output the images to")
    ap.add_argument('--log-dir', type=str, required=True,
                    help="Path to the folder to output the job logs to")
    ap.add_argument('--cpu-config-file', type=str, default='',
                    help="Path to the CPU config file")
    ap.add_argument('--data-shape', type=lambda s: list(map(int, s.split(','))), default=[128, 128, 128],
                    help="Comma separated date Year,Month,Day")
    ap.add_argument('--num-timepoints-per-image', type=int, default=16,
                    help="Number of timepoints in a training image")
    args = ap.parse_args()
    input_file = args.input_file
    output_folder = args.output_folder
    log_dir = args.log_dir
    cpu_config_file = args.cpu_config_file
    data_shape = args.data_shape
    batch_size = args.num_timepoints_per_image
    data_shape.insert(0, batch_size)
    date_ymd = [str(datetime.now().year), str(datetime.now().month), str(datetime.now().day)]
    run_decon_dsr = False
    decon_dsr_jobs = {}
    decon_dsr_job_start_times = {}
    decon_dsr_job_times = {}

    with open(input_file) as f:
        datasets = json.load(f)

    with open(inspect.getfile(XR_decon_data_wrapper), "r", encoding="utf-8") as f:
        decon_dsr_text = f.read()

    with open(inspect.getfile(XR_deskew_rotate_data_wrapper), "r", encoding="utf-8") as f:
        decon_dsr_text += f.read()

    matches = re.findall(r'"(.*?)": \[', decon_dsr_text)
    valid_params = {key: True for key in matches}

    for folder_path, dataset in datasets.items():
        if dataset.get('decon') or dataset.get('dsr'):
            run_decon_dsr = True
            for param, value in dataset.items():
                if param == 'decon' or param == 'dsr':
                    continue
                if param not in valid_params:
                    raise SystemExit(f'{param} is not a valid parameter for PetaKit5D!')

            for i, channel_pattern in enumerate(dataset['channelPatterns']):
                file_count = len(glob.glob(f'{folder_path}/*{channel_pattern}*.tif'))
                num_images_per_dataset = math.floor(file_count / batch_size)
                if not num_images_per_dataset:
                    raise SystemExit(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                     f'Found {file_count} files and the batch size is {batch_size}.')
                datasets[folder_path]['input_is_zarr'] = True
                script = create_decon_dsr_sbatch_script(input_file, folder_path, channel_pattern, dataset.get('decon'),
                                                        dataset.get('dsr'),
                                                        log_dir)
                key = f'Folder: {folder_path} Channel: {channel_pattern}'
                decon_dsr_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE)
                decon_dsr_job.stdin.write(script)
                decon_dsr_job.stdin.close()

                decon_dsr_job_start_times[key] = time.time()
                decon_dsr_jobs[key] = decon_dsr_job
    if run_decon_dsr:
        print('Waiting for Decon/DSR jobs to finish')
        while decon_dsr_jobs:
            for key, decon_dsr_job in decon_dsr_jobs.copy().items():
                if decon_dsr_job.poll() is not None:
                    decon_dsr_job_times[key] = time.time() - decon_dsr_job_start_times[key]
                    print(f'Job for {key} Finished!')
                    del decon_dsr_jobs[key]
            time.sleep(1)
        print('All Decon/DSR jobs done!')

    training_image_jobs = {}
    folders_to_delete = []
    elapsed_sec = 0
    curr_training_image_num = 0
    zarr_channel_patterns = {}
    for folder_path, dataset in datasets.items():
        ext = 'tif'
        if dataset.get('input_is_zarr'):
            ext = 'zarr'
        orig_folder_path = folder_path
        metadata = dataset.copy()
        metadata['input_folder'] = orig_folder_path
        metadata['output_folder'] = str(os.path.join(output_folder, *date_ymd, os.path.basename(orig_folder_path)))
        metadata['software_version'] = f'PyPetaKit5D {version("PyPetaKit5D")}'
        metadata['training_images'] = {}
        if dataset.get('decon'):
            if 'resultDirName' in dataset:
                folder_path = os.path.join(folder_path, dataset['resultDirName'])
            else:
                folder_path = os.path.join(folder_path, 'matlab_decon')
            folders_to_delete.append(folder_path)
        if dataset.get('dsr'):
            if 'DSRDirName' in dataset:
                folder_path = os.path.join(folder_path, dataset['DSRDirName'])
            else:
                folder_path = os.path.join(folder_path, 'DSR')
            if not dataset.get('decon'):
                folders_to_delete.append(folder_path)

        # Create new Channel Patterns based on tiles
        channel_patterns_copy = copy.deepcopy(dataset['channelPatterns'])
        channel_patterns = set()
        # Define the pattern for chunked files
        tiled = True
        pattern = re.compile(r'\d+x_\d+y_\d+z')
        for channel_pattern in channel_patterns_copy:
            all_files = [
                f for f in Path(folder_path).glob('*' + channel_pattern + f'*.{ext}')
            ]
            # Extract matching patterns and add to the set
            for f in all_files:
                if match := pattern.search(f.name):
                    channel_patterns.add(f'{channel_pattern}*{match.group(0)}')
        if channel_patterns:
            channel_patterns = sorted(channel_patterns)
            dataset['channelPatterns'] = channel_patterns
        else:
            channel_patterns = channel_patterns_copy
            tiled = False
        num_orig_patterns = int(len(channel_patterns_copy))
        curr_channel = -1
        orig_channel_patterns = {}
        zarr_channel_pattern = ''
        curr_data_shape = copy.deepcopy(data_shape)
        curr_chunk_shape = copy.deepcopy(data_shape)
        curr_data_shape.insert(0, 1)
        curr_data_shape.append(num_orig_patterns)
        curr_chunk_shape.insert(0, 1)
        curr_chunk_shape.append(1)
        for channel_pattern in dataset['channelPatterns']:
            orig_channel_pattern = copy.deepcopy(channel_pattern)
            if tiled:
                orig_channel_pattern = orig_channel_pattern.split('*', 1)[0]
            if dataset.get('decon') or dataset.get('dsr'):
                elapsed_sec = decon_dsr_job_times[f'Folder: {orig_folder_path} Channel: {orig_channel_pattern}']
            else:
                elapsed_sec = 0

            files = glob.glob(f'{folder_path}/*{channel_pattern}*.{ext}')
            file_count = len(files)
            num_images_per_dataset = math.floor(file_count / batch_size)
            if not num_images_per_dataset:
                raise Exception(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                f'Found {file_count} files and the batch size is {batch_size}.')
            bboxes = get_chunk_bboxes(folder_path, files[0], (batch_size, 128, 128, 128), dataset.get('input_is_zarr'))
            num_chunks_per_image = len(bboxes)

            if not orig_channel_pattern in orig_channel_patterns:
                orig_channel_patterns[orig_channel_pattern] = True
                curr_channel += 1
                curr_data_shape[0] = num_images_per_dataset*num_chunks_per_image

            zarr_channel_pattern = f'{channel_pattern}.zarr'
            if tiled:
                zarr_channel_pattern = f'{channel_pattern.split("*", 1)[1]}.zarr'
            if not zarr_channel_pattern in zarr_channel_patterns:
                zarr_channel_patterns[zarr_channel_pattern] = True
                zarr_out_folder = str(
                    os.path.join(output_folder, *date_ymd, os.path.basename(metadata['input_folder'])))
                zarr_spec = {
                    'driver': 'zarr',
                    'kvstore': {
                        'driver': 'file',
                        'path': f'{os.path.join(os.path.normpath(zarr_out_folder), zarr_channel_pattern)}'
                    },
                    'metadata': {
                        'dtype': '<u2',
                        'shape': curr_data_shape,
                        'chunks': curr_chunk_shape,
                        'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
                        'fill_value': 0,
                        'order': 'C'
                    },
                    'create': True,
                    "delete_existing": True
                }
                ts.open(zarr_spec).result()

            batch_start_number = 0
            input_is_zarr_filenames = dataset.get('input_is_zarr')
            if dataset.get('decon') or dataset.get('dsr'):
                input_is_zarr_filenames = False
            filenames = get_filenames(orig_folder_path, channel_pattern, input_is_zarr_filenames)
            metadata['elapsed_sec'] = elapsed_sec
            for i in range(num_images_per_dataset):
                script = create_file_conversion_sbatch_script(input_file, folder_path, channel_pattern, tiled, curr_channel, output_folder,
                                                              (i * num_chunks_per_image), i * batch_size,
                                                              batch_size, dataset.get('input_is_zarr'), date_ymd,
                                                              elapsed_sec, orig_folder_path, log_dir)
                key = f'Folder: {folder_path} Channel: {channel_pattern}'
                training_image_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE)
                training_image_job.stdin.write(script)
                training_image_job.stdin.close()
                training_image_jobs[key] = training_image_job

                #metadata_filenames = ','.join(filenames[i * batch_size:(i * batch_size) + batch_size])
                metadata_filenames = filenames[i * batch_size:(i * batch_size) + batch_size]
                if not metadata['training_images'].get(zarr_channel_pattern):
                    metadata['training_images'][zarr_channel_pattern] = {}
                if not metadata['training_images'][zarr_channel_pattern].get('channelPatterns'):
                    metadata['training_images'][zarr_channel_pattern]['channelPatterns'] = []
                metadata['training_images'][zarr_channel_pattern]['channelPatterns'].append(orig_channel_pattern)
                metadata['training_images'][zarr_channel_pattern]['filenames'] = metadata_filenames
                if not metadata['training_images'][zarr_channel_pattern].get('chunk_names'):
                    metadata['training_images'][zarr_channel_pattern]['chunk_names'] = {}
                for j in range(num_chunks_per_image):
                    filename = f'{((i * num_chunks_per_image) + j)}.0.0.0.0.{curr_channel}'
                    metadata['training_images'][zarr_channel_pattern]['chunk_names'][filename] = {}
                    metadata['training_images'][zarr_channel_pattern]['chunk_names'][filename]['bbox'] = bboxes[j]
                curr_training_image_num += num_chunks_per_image
                datasets[orig_folder_path]['num_chunks_per_image'] = num_chunks_per_image
        datasets[orig_folder_path]['metadata'] = metadata
        datasets[orig_folder_path]['num_training_images'] = curr_training_image_num

    print('Waiting for Training image creation jobs to finish')
    while training_image_jobs:
        for key, training_image_job in training_image_jobs.copy().items():
            if training_image_job.poll() is not None:
                print(f'Job for {key} Finished!')
                del training_image_jobs[key]
        time.sleep(1)
    print('All Training image creation jobs done!')

    # Matlab processing
    for folder_path, dataset in datasets.items():
        for zarr_filename, training_images in datasets[folder_path]['metadata']['training_images'].items():
            training_image = os.path.join(datasets[folder_path]['metadata']['output_folder'], zarr_filename)
            for chunk_name, chunk_names in training_images['chunk_names'].items():
                split_chunk_name = chunk_name.split('.')
                channel_i = int(split_chunk_name[5])
                script = create_sbatch_matlab_script(training_image, datasets[folder_path]['metadata']['psfFullpaths'][channel_i],
                                                     split_chunk_name[0], split_chunk_name[1], channel_i)
                print(script)


    for folder_path, dataset in datasets.items():
        metadata_object = json.dumps(datasets[folder_path]['metadata'], indent=4)
        with open(
                f'{os.path.normpath(os.path.join(datasets[folder_path]["metadata"]["output_folder"], "metadata"))}.json',
                'w') as outfile:
            outfile.write(metadata_object)

    if folders_to_delete:
        print('Cleaning up intermediate results')
        for folder_to_delete in folders_to_delete:
            shutil.rmtree(folder_to_delete)
        print('Done!')

import argparse
import glob
import subprocess
import math
import json
import os
import sys
import shutil
from convert_files import get_chunk_bboxes
import inspect
import re
from PyPetaKit5D import XR_decon_data_wrapper
from PyPetaKit5D import XR_deskew_rotate_data_wrapper


def create_sbatch_script(python_script_name, folder_path, channel_pattern, output_folder='', output_name_start_num=0, batch_start_number=0,
                         batch_size=16, input_is_zarr=False, log_dir='', decon=False, dsr=False, input_file=''):
    extra_params = ''
    cpus_per_task = 24
    if python_script_name == 'convert_files.py':
        extra_params = f'''--output-folder {output_folder} --output-name-start-num {output_name_start_num} --batch-start-number {batch_start_number} --batch-size {batch_size} '''
        if input_is_zarr:
            extra_params += '--input-is-zarr '
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


def create_file_conversion_sbatch_script(folder_path, channel_pattern, output_folder, output_name_start_num, batch_start_number, batch_size,
                                         input_is_zarr, log_dir):
    return create_sbatch_script('convert_files.py', folder_path, channel_pattern, output_folder=output_folder,
                                output_name_start_num=output_name_start_num, batch_start_number=batch_start_number, batch_size=batch_size,
                                input_is_zarr=input_is_zarr,
                                log_dir=log_dir)


def create_decon_dsr_sbatch_script(folder_path, channel_pattern, decon, dsr, input_file, log_dir):
    if decon is None:
        decon = False
    if dsr is None:
        dsr = False
    return create_sbatch_script('decon_dsr.py', folder_path, channel_pattern, decon=decon, dsr=dsr,
                                input_file=input_file, log_dir=log_dir)


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
    args = ap.parse_args()
    input_file = args.input_file
    output_folder = args.output_folder
    log_dir = args.log_dir
    cpu_config_file = args.cpu_config_file
    run_decon_dsr = False
    batch_size = 8
    decon_dsr_jobs = []

    with open(input_file) as f:
        datasets = json.load(f)

    with open(inspect.getfile(XR_decon_data_wrapper), "r", encoding="utf-8") as f:
        decon_dsr_text = f.read()

    with open(inspect.getfile(XR_deskew_rotate_data_wrapper), "r", encoding="utf-8") as f:
        decon_dsr_text += f.read()

    matches = re.findall(r'"(.*?)": \[', decon_dsr_text)
    valid_params = {key: True for key in matches}

    for folder_path, dataset in datasets.items():
        if 'decon' in dataset or 'dsr' in dataset:
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
                script = create_decon_dsr_sbatch_script(folder_path, channel_pattern, dataset.get('decon'), dataset.get('dsr'),
                                                        input_file, log_dir)

                decon_dsr_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                 stdout=subprocess.PIPE,
                                                 stderr=subprocess.PIPE)
                decon_dsr_job.stdin.write(script)
                decon_dsr_jobs.append(decon_dsr_job)
    if run_decon_dsr:
        print('Waiting for Decon/DSR jobs to finish')
        for decon_dsr_job in decon_dsr_jobs:
            stdout, stderr = decon_dsr_job.communicate()
            print(f'Job {stdout.strip().split()[-1]} Finished!')
        print('All Decon/DSR jobs done!')

    training_image_jobs = []
    folders_to_delete = []
    for folder_path, dataset in datasets.items():
        if dataset.get('decon'):
            folder_path += '/matlab_decon_omw/'
            folders_to_delete.append(folder_path)
        if dataset.get('dsr'):
            folder_path += '/DSR/'
            if not dataset.get('decon'):
                folders_to_delete.append(folder_path)
        for channel_pattern in dataset['channelPatterns']:
            batch_start_number = 0
            ext = 'tif'
            if dataset.get('input_is_zarr'):
                ext = 'zarr'
            files = glob.glob(f'{folder_path}/*{channel_pattern}*.{ext}')
            file_count = len(files)
            num_images_per_dataset = math.floor(file_count / batch_size)
            num_chunks_per_image = len(get_chunk_bboxes(folder_path, files[0], (128,128,128,batch_size), dataset.get('input_is_zarr')))
            if not num_images_per_dataset:
                raise Exception(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                f'Found {file_count} files and the batch size is {batch_size}.')
            #get_filenames(folder_path, channel_pattern, dataset.input_is_zarr)
            #num_training_images = 1
            for i in range(num_images_per_dataset):
                script = create_file_conversion_sbatch_script(folder_path, channel_pattern, output_folder,
                                                              i * num_chunks_per_image, i * batch_size,
                                                              batch_size, dataset.get('input_is_zarr'), log_dir)
                training_image_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE)
                training_image_job.stdin.write(script)
                training_image_jobs.append(training_image_job)

    print('Waiting for Training image creation jobs to finish')
    for training_image_job in training_image_jobs:
        stdout, stderr = training_image_job.communicate()
        print(f'Job {stdout.strip().split()[-1]} Finished!')
    print('All Training image creation jobs done!')
    if folders_to_delete:
        print('Cleaning up intermediate results')
        for folder_to_delete in folders_to_delete:
            shutil.rmtree(folder_to_delete)
        print('Done!')

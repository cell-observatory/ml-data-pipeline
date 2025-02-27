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
from convert_files import get_chunk_bboxes, get_filenames
import inspect
import re
from datetime import datetime
import time
from PyPetaKit5D import XR_decon_data_wrapper
from PyPetaKit5D import XR_deskew_rotate_data_wrapper


def create_sbatch_script(python_script_name, input_file, folder_path, channel_pattern, output_folder='', output_name_start_num=0, batch_start_number=0,
                         batch_size=16, input_is_zarr=False, date_ymd=None, elapsed_sec=0, orig_folder_path=None, log_dir='', decon=False, dsr=False):
    extra_params = ''
    cpus_per_task = 24
    if python_script_name == 'convert_files.py':
        extra_params = f'''--input-file {input_file} --output-folder {output_folder} --output-name-start-num {output_name_start_num} --batch-start-number {batch_start_number} --batch-size {batch_size} --elapsed-sec {elapsed_sec} '''
        if input_is_zarr:
            extra_params += '--input-is-zarr '
        if date_ymd:
            extra_params += f'--date {",".join(map(str, date_ymd)) } '
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


def create_file_conversion_sbatch_script(input_file, folder_path, channel_pattern, output_folder, output_name_start_num, batch_start_number, batch_size,
                                         input_is_zarr, date_ymd, elapsed_sec, orig_folder_path, log_dir):
    return create_sbatch_script('convert_files.py', input_file, folder_path, channel_pattern, output_folder=output_folder,
                                output_name_start_num=output_name_start_num, batch_start_number=batch_start_number, batch_size=batch_size,
                                input_is_zarr=input_is_zarr, date_ymd=date_ymd, elapsed_sec=elapsed_sec, orig_folder_path=orig_folder_path, log_dir=log_dir)


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
    args = ap.parse_args()
    input_file = args.input_file
    output_folder = args.output_folder
    log_dir = args.log_dir
    cpu_config_file = args.cpu_config_file
    date_ymd = [str(datetime.now().year), str(datetime.now().month), str(datetime.now().day)]
    run_decon_dsr = False
    batch_size = 8
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
                script = create_decon_dsr_sbatch_script(input_file, folder_path, channel_pattern, dataset.get('decon'), dataset.get('dsr'),
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

    training_image_jobs = []
    folders_to_delete = []
    elapsed_sec = 0
    curr_training_image_num = 0
    for folder_path, dataset in datasets.items():
        orig_folder_path = folder_path
        metadata = dataset.copy()
        if dataset.get('decon'):
            if 'resultDirName' in dataset:
                folder_path = os.path.join(folder_path, dataset['resultDirName'])
            else:
                folder_path = os.path.join(folder_path, 'matlab_decon')
            #folder_path += '/matlab_decon/'
            folders_to_delete.append(folder_path)
        if dataset.get('dsr'):
            if 'DSRDirName' in dataset:
                folder_path = os.path.join(folder_path, dataset['DSRDirName'])
            else:
                folder_path = os.path.join(folder_path, 'DSR')
            #folder_path += '/DSR/'
            if not dataset.get('decon'):
                folders_to_delete.append(folder_path)
        for channel_pattern in dataset['channelPatterns']:
            if dataset.get('decon') or dataset.get('dsr'):
                elapsed_sec = decon_dsr_job_times[f'Folder: {orig_folder_path} Channel: {channel_pattern}']
            else:
                elapsed_sec = 0
            batch_start_number = 0
            ext = 'tif'
            if dataset.get('input_is_zarr'):
                ext = 'zarr'
            files = glob.glob(f'{folder_path}/*{channel_pattern}*.{ext}')
            file_count = len(files)
            num_images_per_dataset = math.floor(file_count / batch_size)
            bboxes = get_chunk_bboxes(folder_path, files[0], (128,128,128,batch_size), dataset.get('input_is_zarr'))
            num_chunks_per_image = len(bboxes)
            input_is_zarr_filenames = dataset.get('input_is_zarr')
            if dataset.get('decon') or dataset.get('dsr'):
                input_is_zarr_filenames = False
            filenames = get_filenames(orig_folder_path, channel_pattern, input_is_zarr_filenames)
            if not num_images_per_dataset:
                raise Exception(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                f'Found {file_count} files and the batch size is {batch_size}.')
            metadata['input_folder'] = orig_folder_path
            metadata['output_folder'] = str(os.path.join(output_folder, *date_ymd, os.path.basename(orig_folder_path)))
            metadata['software_version'] = f'PyPetaKit5D {version("PyPetaKit5D")}'
            metadata['elapsed_sec'] = elapsed_sec
            for i in range(num_images_per_dataset):
                script = create_file_conversion_sbatch_script(input_file, folder_path, channel_pattern, output_folder,
                                                              (i * num_chunks_per_image), i * batch_size,
                                                              batch_size, dataset.get('input_is_zarr'), date_ymd,
                                                              elapsed_sec, orig_folder_path, log_dir)
                training_image_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE)
                training_image_job.stdin.write(script)
                training_image_jobs.append(training_image_job)
                metadata['training_images'] = {}
                metadata_filenames = ','.join(filenames[i * batch_size:(i * batch_size) + batch_size])
                metadata['training_images'][metadata_filenames] = {}
                metadata['training_images'][metadata_filenames]['channelPatterns'] = [channel_pattern]
                metadata['training_images'][metadata_filenames]['filenames'] = {}
                for j in range(num_chunks_per_image):
                    filename = f'{((i * num_chunks_per_image) + j)}.zarr'
                    metadata['training_images'][metadata_filenames]['filenames'][filename] = {}
                    metadata['training_images'][metadata_filenames]['filenames'][filename]['bbox'] = bboxes[j]
                curr_training_image_num += num_chunks_per_image
                datasets[orig_folder_path]['num_chunks_per_image'] = num_chunks_per_image
        datasets[orig_folder_path]['metadata'] = metadata
        datasets[orig_folder_path]['num_training_images'] = curr_training_image_num


    print('Waiting for Training image creation jobs to finish')
    for training_image_job in training_image_jobs:
        stdout, stderr = training_image_job.communicate()
        print(f'Job {stdout.strip().split()[-1]} Finished!')
    print('All Training image creation jobs done!')
    datasets_copy = copy.deepcopy(datasets)
    for folder_path, dataset in datasets_copy.items():
        files = glob.glob(f'{dataset["metadata"]["output_folder"]}/*.zarr')
        files_sorted = sorted(files, key=lambda x: int(os.path.basename(x).split('.')[0]))
        num_files = len(files_sorted)
        if num_files < dataset['num_training_images']:
            i = 0
            j = 1
            for filenames_str, training_image_info in dataset['metadata']['training_images'].items():
                datasets[folder_path]['metadata']['training_images'][filenames_str]['filenames'] = {}
                for filename, zarr_info in training_image_info['filenames'].items():
                    if i >= num_files:
                        break
                    #curr_num = int(filename.split('.')[0])
                    curr_file_num = int(os.path.basename(files_sorted[i]).split('.')[0])
                    if curr_file_num > dataset['num_chunks_per_image']*(j):
                        j += 1
                        break
                    datasets[folder_path]['metadata']['training_images'][filenames_str]['filenames'][f'{i}.zarr'] = training_image_info['filenames'][f'{curr_file_num}.zarr']
                    os.rename(files_sorted[i], os.path.join(dataset['metadata']['output_folder'], f'{i}.zarr'))
                    i += 1
        metadata_object = json.dumps(datasets[folder_path]['metadata'], indent=4)
        with open(f'{os.path.normpath(os.path.join(datasets[folder_path]["metadata"]["output_folder"], "metadata"))}.json', 'w') as outfile:
            outfile.write(metadata_object)
    if folders_to_delete:
        print('Cleaning up intermediate results')
        for folder_to_delete in folders_to_delete:
            shutil.rmtree(folder_to_delete)
        print('Done!')

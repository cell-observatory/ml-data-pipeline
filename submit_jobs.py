import argparse
import glob
import subprocess
import math
import csv
import json
import os
import sys
import shutil


def create_sbatch_script(python_script_name, folder_path, channel_pattern, output_folder='', output_name_start_num=0,
                         batch_size=16, input_is_zarr=False, log_dir='', decon=False, dsr=False, psf_full_paths='',
                         cpu_config_file=None):
    extra_params = ''
    cpus_per_task = 2
    if python_script_name == 'convert_files.py':
        extra_params = f'''--output-folder {output_folder} --output-name-start-num {output_name_start_num} --batch-size {batch_size} '''
        if input_is_zarr:
            extra_params += '--input-is-zarr '
    elif python_script_name == 'decon_dsr.py':
        cpus_per_task = 24
        if decon:
            extra_params += '--decon '
            extra_params += f'''--psf-full-paths {','.join(psf_full_paths)} '''
        if dsr:
            extra_params += '--dsr '
        if cpu_config_file:
            extra_params += f'--cpu-config-file {cpu_config_file} '
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


def create_file_conversion_sbatch_script(folder_path, channel_pattern, output_folder, output_name_start_num, batch_size,
                                         input_is_zarr, log_dir):
    return create_sbatch_script('convert_files.py', folder_path, channel_pattern, output_folder=output_folder,
                                output_name_start_num=output_name_start_num, batch_size=batch_size,
                                input_is_zarr=input_is_zarr,
                                log_dir=log_dir)


def create_decon_dsr_sbatch_script(folder_path, channel_pattern, decon, dsr, psf_full_paths, cpu_config_file, log_dir):
    return create_sbatch_script('decon_dsr.py', folder_path, channel_pattern, decon=decon, dsr=dsr,
                                psf_full_paths=psf_full_paths, cpu_config_file=cpu_config_file, log_dir=log_dir)


class Dataset:
    def __init__(self, channel_patterns, decon, dsr, psf_full_paths):
        self.channel_patterns = channel_patterns
        self.decon = json.loads(decon.lower())
        self.dsr = json.loads(dsr.lower())
        self.psf_full_paths = psf_full_paths
        self.input_is_zarr = False


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Path to the csv file containing dataset information")
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
    datasets = {}
    run_decon_dsr = False
    with open(input_file) as file:
        reader = csv.DictReader(file)
        for row in reader:
            row: dict
            dataset = Dataset(row.get('channel_patterns').split(), row.get('decon'), row.get('dsr'),
                              row.get('psf_full_paths').split())
            if dataset.decon or dataset.dsr:
                run_decon_dsr = True
            datasets[row['folder']] = dataset
    batch_size = 8
    decon_dsr_jobs = []

    if run_decon_dsr:
        for folder_path, dataset in datasets.items():
            for i, channel_pattern in enumerate(dataset.channel_patterns):
                file_count = len(glob.glob(f'{folder_path}/*{channel_pattern}*.tif'))
                num_images_per_dataset = math.floor(file_count / batch_size)
                if not num_images_per_dataset:
                    raise Exception(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                    f'Found {file_count} files and the batch size is {batch_size}.')
                num_training_images = 1
                num_runs = math.ceil(num_training_images / num_images_per_dataset)
                for j in range(num_runs):
                    if not dataset.decon and not dataset.dsr:
                        continue
                    datasets[folder_path].input_is_zarr = True
                    script = create_decon_dsr_sbatch_script(folder_path, channel_pattern, dataset.decon, dataset.dsr,
                                                            [dataset.psf_full_paths[i]], cpu_config_file, log_dir)

                    decon_dsr_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                     stdout=subprocess.PIPE,
                                                     stderr=subprocess.PIPE)
                    decon_dsr_job.stdin.write(script)
                    decon_dsr_jobs.append(decon_dsr_job)
        print('Waiting for Decon/DSR jobs to finish')
        for decon_dsr_job in decon_dsr_jobs:
            stdout, stderr = decon_dsr_job.communicate()
            print(f'Job {stdout.strip().split()[-1]} Finished!')
        print('All Decon/DSR jobs done!')

    training_image_jobs = []
    folders_to_delete = []
    for folder_path, dataset in datasets.items():
        if dataset.decon:
            folder_path += '/matlab_decon_omw/'
            folders_to_delete.append(folder_path)
        if dataset.dsr:
            folder_path += '/DSR/'
            if not dataset.decon:
                folders_to_delete.append(folder_path)
        for channel_pattern in dataset.channel_patterns:
            ext = 'tif'
            if dataset.input_is_zarr:
                ext = 'zarr'
            file_count = len(glob.glob(f'{folder_path}/*{channel_pattern}*.{ext}'))
            num_images_per_dataset = math.floor(file_count / batch_size)
            if not num_images_per_dataset:
                raise Exception(f'Not enough files in {folder_path} for channel pattern \'{channel_pattern}\'!\n'
                                f'Found {file_count} files and the batch size is {batch_size}.')
            num_training_images = 1
            num_runs = math.ceil(num_training_images / num_images_per_dataset)
            for i in range(num_runs):
                script = create_file_conversion_sbatch_script(folder_path, channel_pattern, output_folder,
                                                              i * num_images_per_dataset,
                                                              batch_size, dataset.input_is_zarr, log_dir)
                training_image_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                      stdout=subprocess.PIPE,
                                                      stderr=subprocess.PIPE)
                training_image_job.stdin.write(script)
                training_image_jobs.append(training_image_job)
                #result = subprocess.run(["sbatch"], input=script, text=True, capture_output=True)
                #print(result.stdout)
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

import argparse
import copy
import glob
import json
import os.path
import re
import subprocess
from pathlib import Path
import numpy as np

from supabase import create_client
import tensorstore as ts

from submit_jobs import create_augmentation_sbatch_script


def get_chunk_bboxes(data_shape, im_shape):
    bboxes = []
    z = 0
    while z + data_shape[1] <= im_shape[0]:
        y = 0
        while y + data_shape[2] <= im_shape[1]:
            x = 0
            while x + data_shape[3] <= im_shape[2]:
                bboxes.append([z, y, x, z + data_shape[1], y + data_shape[2], x + data_shape[3]])
                x += data_shape[3]
            y += data_shape[2]
        z += data_shape[1]
    return bboxes

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing zarr files separated by comma")
    ap.add_argument('--server-folder', type=str, required=True,
                    help="Path to where the actual data folder begins")
    ap.add_argument('--url', type=str, required=True,
                    help="Database URL")
    ap.add_argument('--key', type=str, required=True,
                    help="Database Key")
    ap.add_argument('--num-timepoints-per-image', type=int, default=16,
                    help="Number of timepoints in a training image")
    args = ap.parse_args()
    folder_paths = args.folder_paths
    server_folder = args.server_folder
    url = args.url
    key = args.key
    num_timepoints_per_image = args.num_timepoints_per_image

    orig_pixel_size = [0.097, 0.097, 0.097]

    pixel_sizes = [[0.097, 0.15, 0.15], [0.097, 0.2, 0.15]]
    angles = [[5, 5], [10, 10]]

    supabase = create_client(url, key)

    for folder_path in folder_paths:
        output_folder = str(Path(folder_path).relative_to(Path(server_folder)))
        with open(f'{os.path.join(folder_path, "metadata.json")}', 'r') as json_file:
            metadata = json.load(json_file)
        i = 0
        augmentation_info = {}
        cube_size = metadata['cube_size']

        tile_paths = glob.glob(f'{folder_path}/*.zarr')
        tile_path = tile_paths[0]
        im_ts = ts.open(
            {
                "driver": "zarr3",
                "kvstore": {
                    "driver": "file",
                    "path": tile_path
                }
            },
            read=True
        ).result()
        num_channels = im_ts.domain.shape[4]
        num_timepoints = im_ts.domain.shape[0]
        im_size = [im_ts.domain.shape[1],im_ts.domain.shape[2],im_ts.domain.shape[3]]

        num_z_chunks = int(im_size[0] / cube_size)
        num_y_chunks = int(im_size[1] / cube_size)
        num_x_chunks = int(im_size[2] / cube_size)
        max_chunks_zyx = max(num_z_chunks, num_y_chunks, num_x_chunks)
        num_chunks = num_z_chunks * num_y_chunks * num_x_chunks

        for pixel_size in pixel_sizes:
            for angle in angles:
                pixel_size_inc = np.divide(pixel_size, orig_pixel_size)
                new_size = np.round(np.multiply(cube_size, pixel_size_inc)).astype(int).tolist()
                if np.any(new_size > np.array(im_size)):
                    continue
                augmentation_info[i] = {}
                augmentation_info[i]['pixel_size'] = pixel_size
                #augmentation_info[i]['pixel_size_inc'] = np.divide(pixel_size, orig_pixel_size)
                augmentation_info[i]['angle'] = angle
                augmentation_info[i]['new_size'] = new_size

                i += 1

        for num, augmentation in augmentation_info.items():
            #bboxes = get_chunk_bboxes([1] + augementation['new_size'].tolist(), im_shape=im_size)
            new_metadata = copy.deepcopy(metadata)
            new_metadata['input_folder'] = os.path.join(new_metadata['server_folder'], new_metadata['output_folder'])
            new_metadata['output_folder'] = os.path.join('augmented',
                                                         f'angle_{augmentation["angle"][0]}z_{augmentation["angle"][1]}xy_'
                                                         f'pixel_size_{str(augmentation["pixel_size"][0]).replace(".", "p")}z_'
                                                         f'{str(augmentation["pixel_size"][1]).replace(".", "p")}y_'
                                                         f'{str(augmentation["pixel_size"][2]).replace(".", "p")}x',
                                                         new_metadata['output_folder'])
            new_metadata['augmentation_info'] = augmentation
            for tile_name, tile in metadata['training_images'].items():
                new_metadata['training_images'][tile_name]['chunk_names'] = {}
                for chunk_name, chunk in tile['chunk_names'].items():
                    if (chunk['bbox'][0]+augmentation['new_size'][0] > tile['bbox'][3] or
                            chunk['bbox'][1]+augmentation['new_size'][1] > tile['bbox'][4] or
                            chunk['bbox'][2]+augmentation['new_size'][2] > tile['bbox'][5]):
                        continue
                    new_metadata['training_images'][tile_name]['chunk_names'][chunk_name] = {}
                    new_metadata['training_images'][tile_name]['chunk_names'][chunk_name]['bbox'] = [chunk['bbox'][0]-tile['bbox'][0], chunk['bbox'][1]-tile['bbox'][1], chunk['bbox'][2]-tile['bbox'][2],
                                                                                                     chunk['bbox'][3]-tile['bbox'][0], chunk['bbox'][4]-tile['bbox'][1], chunk['bbox'][5]-tile['bbox'][2]]
                    new_metadata['training_images'][tile_name]['chunk_names'][chunk_name]['augmentation_bbox'] = [chunk['bbox'][0], chunk['bbox'][1],
                                                                                         chunk['bbox'][2], chunk['bbox'][0]+augmentation['new_size'][0],
                                                                                         chunk['bbox'][1]+augmentation['new_size'][1], chunk['bbox'][2]+augmentation['new_size'][2]]
            metadata_object = json.dumps(new_metadata, indent=4, sort_keys=True)
            dir_path = os.path.normpath(os.path.join(new_metadata["server_folder"], new_metadata["output_folder"]))
            os.makedirs(dir_path, exist_ok=True)
            with open(
                    f'{os.path.normpath(os.path.join(dir_path, "metadata"))}.json',
                    'w') as outfile:
                outfile.write(metadata_object)
            num_images = num_timepoints/num_timepoints_per_image
            for tile_name, tile in new_metadata['training_images'].items():
                for channel_num in range(num_channels):
                    for timepoint_num in range(num_timepoints):
                        for chunk_num in range(0, num_chunks, max_chunks_zyx):
                            script = create_augmentation_sbatch_script(folder_path, tile_name, chunk_num, chunk_num+max_chunks_zyx, timepoint_num, channel_num)
                            '''
                            key = f'Folder: {folder_path} Timepoint: {i} Channel: {channel_pattern}'
                            training_image_job = subprocess.Popen(['sbatch', '--wait'], stdin=subprocess.PIPE, text=True,
                                                                  stdout=subprocess.PIPE,
                                                                  stderr=subprocess.PIPE)
                            training_image_job.stdin.write(script)
                            training_image_job.stdin.close()
                            training_image_jobs[key] = training_image_job
                            '''


import copy
import os
import re
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
import numpy as np
import tensorstore as ts
import cpptiff
import zarr
import json


#from importlib.metadata import version


def extract_number(filename, pattern):
    match = pattern.search(filename)
    return int(match.group(1)) if match else float('inf')


def get_filenames(folder_path, channel_pattern='', input_is_zarr=False):
    ext = 'tif'
    if input_is_zarr:
        ext = 'zarr'
    files = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, f'*{channel_pattern}*.{ext}'))]

    pattern = re.compile(r'_(\d+)msecAbs_')
    sorted_files = sorted(files, key=lambda f: extract_number(f, pattern))

    return sorted_files


def get_chunk_bboxes(folder_path, filename, data_shape, input_is_zarr):
    if not input_is_zarr:
        im_shape = cpptiff.get_image_shape(os.path.join(folder_path, filename))
    else:
        # Bug with ts?
        '''
        im_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {
                'driver': "file",
                'path': f'{os.path.normpath(os.path.join(folder_path, filename))}'
            }
        }).result()
        '''
        im_zarr = zarr.open(f'{os.path.normpath(os.path.join(folder_path, filename))}', mode="r")
        im_shape = (im_zarr.shape[2], im_zarr.shape[0], im_zarr.shape[1])

    bboxes = []
    z = im_shape[0] % data_shape[1] // 2
    while z + data_shape[1] < im_shape[0]:
        y = im_shape[1] % data_shape[2] // 2
        while y + data_shape[2] < im_shape[1]:
            x = im_shape[2] % data_shape[3] // 2
            while x + data_shape[3] < im_shape[2]:
                bboxes.append([z, y, x, z + data_shape[1], y + data_shape[2], x + data_shape[3]])
                x += data_shape[3]
            y += data_shape[2]
        z += data_shape[1]
    return bboxes


def write_zarr_chunks(args):
    out_folder, out_name, data, dataset, date, occ_ratios = args
    occ_ratio = np.mean(occ_ratios)
    data = data[..., np.newaxis]
    zarr_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{os.path.normpath(out_folder)}.zarr'
        },
        'create': False,
    }

    zarr_file = ts.open(zarr_spec).result()
    #zarr_file.write(data).result()
    zarr_file[out_name, ...] = data
    '''
    dataset['output_folder'] = out_folder
    dataset['training_image_filenames'] = filenames
    dataset['training_image_bbox'] = bbox
    dataset['software_version'] = f'PyPetaKit5D {version("PyPetaKit5D")}'
    dataset['elapsed_sec'] = elapsed_sec
    dataset['channelPatterns'] = [channel_pattern]

    json_object = json.dumps(dataset, indent=4)

    with open(f'{os.path.normpath(os.path.join(out_folder, str(out_name)))}.json', 'w') as outfile:
        outfile.write(json_object)
    '''


def process_image(args):
    """Function to read and process an image."""
    index, filename, folder_path, input_is_zarr, data_shape, bboxes, num_bboxes, remove_background = args
    if not input_is_zarr:
        #im = cpptiff.read_tiff(os.path.join(folder_path, filename), [0, bboxes[-1][1]])
        im = cpptiff.read_tiff(os.path.join(folder_path, filename))
    else:
        im_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {
                'driver': "file",
                'path': f'{os.path.normpath(os.path.join(folder_path, filename))}'
            }
        }).result()

        # Data is YXZ for zarr so transpose it after
        #im = im_ts[:bboxes[-1][3], :bboxes[-1][5], :bboxes[-1][1]].read().result()
        im = im_ts.read().result()
        im = np.transpose(im, (2, 0, 1))

    # Remove background if needed
    if remove_background:
        nstddevs = 2
        im = np.clip(im - 100 - (np.std(im) * nstddevs), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    chunks = np.zeros(tuple(data_shape[-3:]) + (num_bboxes,), dtype=np.uint16, order='F')
    min_val = np.percentile(im, 0.1)
    max_val = np.percentile(im, 99.9)
    occ_ratios = np.zeros(num_bboxes, dtype=np.float32, order='F')
    for i, bbox in enumerate(bboxes):
        chunk = im[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        within_bounds = (chunk > min_val) & (chunk < max_val)
        occ_ratios[i] = np.mean(within_bounds)
        chunks[:, :, :, i] = chunk

    '''
    elif im.itemsize > 2:
        im = np.clip(im, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    '''
    return index, chunks, occ_ratios  # Return index and processed image


def convert_tiff_to_zarr(dataset, folder_path, channel_pattern, filenames, out_folder, out_name, batch_size,
                         input_is_zarr, date, channel_num,
                         data_shape=None,
                         remove_background=False):
    if not data_shape:
        data_shape = [batch_size, 128, 128, 128]
    bboxes = get_chunk_bboxes(folder_path, filenames[0], data_shape, input_is_zarr)
    num_bboxes = len(bboxes)

    data = np.zeros(tuple(data_shape) + (len(bboxes),), dtype=np.uint16, order='F')
    num_files = len(filenames)
    occ_ratios = np.zeros((num_files, num_bboxes), dtype=np.float32, order='F')
    args_list = [(i, filenames[i], folder_path, input_is_zarr, data_shape, bboxes, num_bboxes, remove_background) for i
                 in
                 range(num_files)]
    with ProcessPoolExecutor() as executor:
        for i, chunks, occ_ratios_i in executor.map(process_image, args_list):
            data[i, :, :, :, :] = chunks
            occ_ratios[i, :] = occ_ratios_i

    data = np.moveaxis(data, -1, 0)
    data = data[..., np.newaxis]
    # Create the Zarr file for the dataset
    out_folder = str(os.path.join(out_folder, *date, os.path.basename(dataset['input_folder'])))
    zarr_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{os.path.join(os.path.normpath(out_folder), channel_pattern)}.zarr'
        }
    }
    zarr_file = ts.open(zarr_spec).result()
    zarr_file[out_name:out_name + num_bboxes, :, :, :, :, channel_num:channel_num + 1] = data

    occ_ratio_means = np.mean(occ_ratios, axis=0)
    occ_ratio_json = {}
    curr_mean = 0
    for i in range(out_name, out_name + num_bboxes):
        occ_ratio_json[f'{i}.0.0.0.0.{channel_num}'] = float(occ_ratio_means[curr_mean])
        curr_mean += 1
    metadata_object = json.dumps(occ_ratio_json, indent=4)
    with open(
            f'{os.path.normpath(os.path.join(os.path.normpath(out_folder), channel_pattern))}_{out_name}img_{channel_num}ch.json',
            'w') as outfile:
        outfile.write(metadata_object)


if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-file', type=str, required=True,
                    help="Path to the json file containing dataset information")
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--orig-folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the original folder containing tiff files separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument("--tiled", action="store_true", help="Use Zarr instead of tiff for input")
    ap.add_argument('--channel-num', type=int, required=True,
                    help="Zarr channel number")
    ap.add_argument('--output-folder', type=str, required=True,
                    help="Folder to write the data to")
    ap.add_argument('--output-name-start-number', type=int, required=True,
                    help="Number to start the output name at")
    ap.add_argument('--batch-start-number', type=int, required=True,
                    help="Number to start the batch of files at")
    ap.add_argument('--batch-size', type=int, required=True,
                    help="Number of timepoints in a training image")
    ap.add_argument("--input-is-zarr", action="store_true", help="Use Zarr instead of tiff for input")
    ap.add_argument('--date', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Comma separated date Year,Month,Day")
    ap.add_argument('--elapsed-sec', type=float, required=True,
                    help="Preprocessing time in seconds")
    args = ap.parse_args()
    input_file = args.input_file
    folder_paths = args.folder_paths
    orig_folder_paths = args.orig_folder_paths
    channel_patterns = args.channel_patterns
    tiled = args.tiled
    channel_num = args.channel_num
    output_folder = args.output_folder
    output_name_start_number = args.output_name_start_number
    batch_start_number = args.batch_start_number
    batch_size = args.batch_size
    input_is_zarr = args.input_is_zarr
    date = args.date
    elapsed_sec = args.elapsed_sec

    with open(input_file) as f:
        datasets_json = json.load(f)

    datasets = {}
    for i, folder_path in enumerate(folder_paths):
        datasets[folder_path] = datasets_json[orig_folder_paths[i]]
        datasets[folder_path]['input_folder'] = orig_folder_paths[i]

    for folder_path in folder_paths:
        for channel_pattern in channel_patterns:
            filenames = get_filenames(folder_path, channel_pattern, input_is_zarr)
            start = time.time()
            if tiled:
                channel_pattern = channel_pattern.split("*", 1)[1]
            #channel_pattern = channel_pattern.replace('*','_')
            filenames_batch = filenames[batch_start_number:batch_start_number + batch_size]
            convert_tiff_to_zarr(datasets[folder_path], folder_path, channel_pattern, filenames_batch, output_folder,
                                 output_name_start_number, batch_size, input_is_zarr, date, channel_num)

            end = time.time()
            print(f"Time taken to run the code was {end - start} seconds")

import os
import re
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import tensorstore as ts
import cpptiff
import zarr
import json


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


def get_image_shape(folder_path, filename, input_is_zarr):
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
    return im_shape


def get_chunk_bboxes(data_shape, folder_path='', filename='', input_is_zarr=True, im_shape=None):
    if not im_shape:
        im_shape = get_image_shape(folder_path, filename, input_is_zarr)

    bboxes = []
    z = im_shape[0] % data_shape[1] // 2
    while z + data_shape[1] <= im_shape[0]:
        y = im_shape[1] % data_shape[2] // 2
        while y + data_shape[2] <= im_shape[1]:
            x = im_shape[2] % data_shape[3] // 2
            while x + data_shape[3] <= im_shape[2]:
                bboxes.append([z, y, x, z + data_shape[1], y + data_shape[2], x + data_shape[3]])
                x += data_shape[3]
            y += data_shape[2]
        z += data_shape[1]
    return bboxes


def process_image(args):
    """Function to read and process an image."""
    index, filename, folder_path, input_is_zarr, bboxes, num_bboxes = args
    z_min = float('inf')
    y_min = float('inf')
    x_min = float('inf')
    z_max = 0
    y_max = 0
    x_max = 0
    for bbox in bboxes:
        z_min = min(z_min, bbox[0])
        y_min = min(y_min, bbox[1])
        x_min = min(x_min, bbox[2])
        z_max = max(z_max, bbox[3])
        y_max = max(y_max, bbox[4])
        x_max = max(x_max, bbox[5])
    if not input_is_zarr:
        im = cpptiff.read_tiff(os.path.join(folder_path, filename), [z_min, z_max])
        im = im[:, y_min:y_max, x_min:x_max]
        #im = cpptiff.read_tiff(os.path.join(folder_path, filename))
    else:
        im_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {
                'driver': "file",
                'path': f'{os.path.normpath(os.path.join(folder_path, filename))}'
            }
        }).result()

        # Data is YXZ for zarr so transpose it after
        im = im_ts[y_min:y_max, x_min:x_max, z_min:z_max].read().result()
        #im = im_ts.read().result()
        im = np.transpose(im, (2, 0, 1))

    min_val = np.percentile(im, 0.1)
    max_val = np.percentile(im, 99.9)
    occ_ratios = np.zeros(num_bboxes, dtype=np.float32, order='F')
    for i, bbox in enumerate(bboxes):
        chunk = im[bbox[0]-z_min:bbox[3]-z_min, bbox[1]-y_min:bbox[4]-y_min, bbox[2]-x_min:bbox[5]-x_min]
        within_bounds = (chunk > min_val) & (chunk < max_val)
        occ_ratios[i] = np.mean(within_bounds)

    return index, im, occ_ratios  # Return index and processed image


def convert_tiff_to_zarr(dataset, folder_path, channel_pattern, filenames, out_folder, out_name, batch_size,
                         input_is_zarr, date, channel_num, timepoint_i, output_zarr_version, outer_data_shape, data_shape=None):
    if not data_shape:
        data_shape = [batch_size, 128, 128, 128]

    bboxes = get_chunk_bboxes(data_shape, folder_path, filenames[0], input_is_zarr)
    num_bboxes = len(bboxes)

    data = np.zeros((batch_size,) + tuple(outer_data_shape[1:4]), dtype=np.uint16, order='F')
    num_files = len(filenames)
    occ_ratios = np.zeros((num_files, num_bboxes), dtype=np.float32, order='F')
    args_list = [(i, filenames[i], folder_path, input_is_zarr, bboxes, num_bboxes) for i
                 in
                 range(num_files)]
    with ProcessPoolExecutor() as executor:
        for i, im, occ_ratios_i in executor.map(process_image, args_list):
            data[i, :, :, :] = im
            occ_ratios[i, :] = occ_ratios_i

    data = data[..., np.newaxis]

    # Create the Zarr file for the dataset
    folder_3 = os.path.basename(dataset['input_folder'])
    folder_2 = os.path.basename(os.path.dirname(dataset['input_folder']))
    folder_1 = os.path.basename(os.path.dirname(os.path.dirname(dataset['input_folder'])))
    out_folder = str(os.path.join(out_folder, *date, folder_1, folder_2, folder_3))
    zarr_spec = {
        'driver': output_zarr_version,
        'kvstore': {
            'driver': 'file',
            'path': f'{os.path.join(os.path.normpath(out_folder), channel_pattern)}.zarr'
        }
    }
    zarr_file = ts.open(zarr_spec).result()

    zarr_file[timepoint_i * batch_size:(timepoint_i + 1) * batch_size, :, :, :, channel_num:channel_num + 1] = data

    occ_ratio_json = {}
    occ_ratios_i = 0
    for i in range((timepoint_i * batch_size), ((timepoint_i + 1) * batch_size)):
        for j in range(num_bboxes):
            if output_zarr_version == 'zarr3':
                filename = f'c/{j}/{i}/0/0/0/{channel_num}'
            else:
                filename = f'{j}.{i}.0.0.0.{channel_num}'
            occ_ratio_json[filename] = float(occ_ratios[occ_ratios_i][j])
        occ_ratios_i += 1
    metadata_object = json.dumps(occ_ratio_json, indent=4)
    with open(
            f'{os.path.normpath(os.path.join(os.path.normpath(out_folder), channel_pattern))}_{out_name}img_{timepoint_i}t_{channel_num}ch.json',
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
    ap.add_argument('--output-zarr-version', type=str, default='zarr3',
                    help="Zarr version to use for output zarr files. Valid values: zarr or zarr3")
    ap.add_argument('--outer-data-shape', type=lambda s: list(map(int, s.split(','))), default=[16, 128, 128, 128, 1],
                    help="Data shape for the 5D Hypercube (size of the zarr file being written to)")
    ap.add_argument('--data-shape', type=lambda s: list(map(int, s.split(','))), default=[16, 128, 128, 128],
                    help="Data shape for the 4D Hypercube")
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
    output_zarr_version = args.output_zarr_version
    data_shape = args.data_shape
    outer_data_shape = args.outer_data_shape

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
            timepoint_i = int(batch_start_number / batch_size)
            convert_tiff_to_zarr(datasets[folder_path], folder_path, channel_pattern, filenames_batch, output_folder,
                                 output_name_start_number, batch_size, input_is_zarr, date, channel_num, timepoint_i,
                                 output_zarr_version, outer_data_shape, data_shape)

            end = time.time()
            print(f"Time taken to run the code was {end - start} seconds")

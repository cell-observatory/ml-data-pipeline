import os
import re
import glob
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import tensorstore as ts
import cpptiff


def extract_number(filename, pattern):
    match = pattern.search(filename)
    return int(match.group(1)) if match else float('inf')


def get_filenames(folder_path, channel_pattern='', input_is_zarr=False):
    ext = 'tif'
    if input_is_zarr:
        ext = 'zarr'
    tiff_files = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, f'*{channel_pattern}*.{ext}'))]

    pattern = re.compile(r'_(\d+)msecAbs_')
    sorted_files = sorted(tiff_files, key=lambda f: extract_number(f, pattern))

    return sorted_files


def get_chunk_bboxes(folder_path, filename, data_shape, input_is_zarr):
    if not input_is_zarr:
        im_shape = cpptiff.get_image_shape(os.path.join(folder_path, filename))
    else:
        im_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {
                'driver': "file",
                'path': f'{os.path.normpath(os.path.join(folder_path, filename))}'
            }
        }).result()
        im_shape = (im_ts.shape[2], im_ts.shape[0], im_ts.shape[1])
    bboxes = []
    z = im_shape[0] % data_shape[0] // 2
    while z + data_shape[0] < im_shape[0]:
        y = im_shape[1] % data_shape[1] // 2
        while y + data_shape[1] < im_shape[1]:
            x = im_shape[2] % data_shape[2] // 2
            while x + data_shape[2] < im_shape[2]:
                bboxes.append([z, z + data_shape[0], y, y + data_shape[1], z, x, x + data_shape[2]])
                x += data_shape[2]
            y += data_shape[1]
        z += data_shape[0]
    return bboxes

def process_image(args):
    """Function to read and process an image."""
    i, filename, folder_path, input_is_zarr, data_shape, remove_background = args
    if not input_is_zarr:
        im = cpptiff.read_tiff(os.path.join(folder_path, filename), [0, data_shape[0]])
        # Crop x and y from the center
        x_start = (im.shape[2] - data_shape[2]) // 2
        y_start = (im.shape[1] - data_shape[1]) // 2
        im = im[:, y_start:y_start + data_shape[1], x_start:x_start + data_shape[2]]
    else:
        im_ts = ts.open({
            'driver': 'zarr',
            'kvstore': {
                'driver': "file",
                'path': f'{os.path.normpath(os.path.join(folder_path, filename))}'
            }
        }).result()
        x_start = (im_ts.shape[1] - data_shape[2]) // 2
        y_start = (im_ts.shape[0] - data_shape[1]) // 2
        # Data is YXZ for zarr so transpose it after
        im = im_ts[y_start:y_start + data_shape[1], x_start:x_start + data_shape[2], 0:data_shape[0]].read().result()
        im = np.transpose(im, (2, 0, 1))

    # Remove background if needed
    if remove_background:
        nstddevs = 2
        im = np.clip(im - 100 - (np.std(im) * nstddevs), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    '''
    elif im.itemsize > 2:
        im = np.clip(im, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    '''
    return i, im  # Return index and processed image


def convert_tiff_to_zarr(folder_path, filenames, out_folder, out_name, batch_size, input_is_zarr, data_shape=None,
                         remove_background=False):
    if not data_shape:
        data_shape = [128, 128, 128, batch_size]
    data = np.zeros(tuple(data_shape), dtype=np.uint16, order='F')
    bboxes = get_chunk_bboxes(folder_path, filenames[0], data_shape, input_is_zarr)
    # Prepare arguments for threading
    args_list = [(i, filenames[i], folder_path, input_is_zarr, data_shape, remove_background) for i in
                 range(len(filenames))]

    # Use threading for parallel execution
    with ProcessPoolExecutor() as executor:
        for i, im in executor.map(process_image, args_list):
            data[:, :, :, i] = im  # Store result in correct position

    zarr_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{out_folder}/{out_name}.zarr'
        },
        'metadata': {
            'dtype': '<u2',
            'shape': data_shape,
            'chunks': data_shape,
            'compressor': {'blocksize': 0, 'clevel': 1, 'cname': 'zstd', 'id': 'blosc', 'shuffle': 1},
            'fill_value': 0,
            'order': 'F'
        },
        'create': True,
        'delete_existing': False
    }

    zarr_file = ts.open(zarr_spec).result()
    zarr_file.write(data).result()


if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing tiff files separated by comma")
    ap.add_argument('--channel-patterns', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Channel patterns separated by comma")
    ap.add_argument('--output-folder', type=str, required=True,
                    help="Folder to write the data to")
    ap.add_argument('--output-name-start-number', type=int, required=True,
                    help="Number to start the output name at")
    ap.add_argument('--batch-size', type=int, required=True,
                    help="Number of timepoints in a training image")
    ap.add_argument("--input-is-zarr", action="store_true", help="Use Zarr instead of tiff for input")
    args = ap.parse_args()
    folder_paths = args.folder_paths
    channel_patterns = args.channel_patterns
    output_folder = args.output_folder
    output_name_start_number = args.output_name_start_number
    batch_size = args.batch_size
    input_is_zarr = args.input_is_zarr
    for folder_path in folder_paths:
        for channel_pattern in channel_patterns:
            filenames = get_filenames(folder_path, channel_pattern, input_is_zarr)
            start = time.time()

            for i in range(0, len(filenames) - len(filenames) % batch_size, batch_size):
                filenames_batch = filenames[i:i + batch_size]
                convert_tiff_to_zarr(folder_path, filenames_batch, output_folder,
                                     int(output_name_start_number + i / batch_size), batch_size, input_is_zarr)

            end = time.time()
            print(f"Time taken to run the code was {end - start} seconds")

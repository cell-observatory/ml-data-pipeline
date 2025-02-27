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
    z = im_shape[0] % data_shape[0] // 2
    while z + data_shape[0] < im_shape[0]:
        y = im_shape[1] % data_shape[1] // 2
        while y + data_shape[1] < im_shape[1]:
            x = im_shape[2] % data_shape[2] // 2
            while x + data_shape[2] < im_shape[2]:
                bboxes.append([z, y, x, z + data_shape[0], y + data_shape[1], x + data_shape[2]])
                x += data_shape[2]
            y += data_shape[1]
        z += data_shape[0]
    return bboxes


def write_zarr_chunks(args):
    out_folder, out_name, data_shape, data, dataset, folder_path, channel_pattern, filenames, bbox, date, elapsed_sec = args
    out_folder = str(os.path.join(out_folder, *date, os.path.basename(dataset['input_folder'])))
    zarr_spec = {
        'driver': 'zarr',
        'kvstore': {
            'driver': 'file',
            'path': f'{os.path.normpath(os.path.join(out_folder, str(out_name)))}.zarr'
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
    index, filename, folder_path, input_is_zarr, data_shape, bboxes, remove_background, bad_chunks = args
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
    chunks = np.zeros(tuple(data_shape[:3]) + (len(bboxes),), dtype=np.uint16, order='F')
    min_val = np.percentile(im, 0.1)
    max_val = np.percentile(im, 99.9)
    for i, bbox in enumerate(bboxes):
        if i in bad_chunks:
            continue
        chunk = im[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        within_bounds = (chunk > min_val) & (chunk < max_val)
        percentage_within = np.mean(within_bounds)
        if percentage_within >= .8:
            chunks[:, :, :, i] = chunk
        else:
            bad_chunks[i] = True

    '''
    elif im.itemsize > 2:
        im = np.clip(im, 0, np.iinfo(np.uint16).max).astype(np.uint16)
    '''
    return index, chunks  # Return index and processed image


def convert_tiff_to_zarr(dataset, folder_path, channel_pattern, filenames, out_folder, out_name, batch_size,
                         input_is_zarr, date, elapsed_sec,
                         data_shape=None,
                         remove_background=False):
    if not data_shape:
        data_shape = [128, 128, 128, batch_size]
    bboxes = get_chunk_bboxes(folder_path, filenames[0], data_shape, input_is_zarr)
    num_bboxes = len(bboxes)

    data = np.zeros(tuple(data_shape) + (len(bboxes),), dtype=np.uint16, order='F')



    with Manager() as manager:
        bad_chunks = manager.dict()
        args_list = [(i, filenames[i], folder_path, input_is_zarr, data_shape, bboxes, remove_background, bad_chunks) for i in
                     range(len(filenames))]
        with ProcessPoolExecutor() as executor:
            for i, chunks in executor.map(process_image, args_list):
                data[:, :, :, i, :] = chunks
                #bad_chunks = {**bad_chunks, **bad_chunks_i}
        bad_chunks = dict(bad_chunks)

    args_list_chunks = [
        (out_folder, out_name + i, data_shape, data[:, :, :, :, i], dataset, folder_path, channel_pattern, filenames,
         bboxes[i], date, elapsed_sec)
        for i in
        range(num_bboxes) if not bad_chunks.get(i, False)]
    with ProcessPoolExecutor() as executor:
        for result in executor.map(write_zarr_chunks, args_list_chunks):
            pass


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

            filenames_batch = filenames[batch_start_number:batch_start_number + batch_size]
            convert_tiff_to_zarr(datasets[folder_path], folder_path, channel_pattern, filenames_batch, output_folder,
                                 output_name_start_number, batch_size, input_is_zarr, date, elapsed_sec)

            end = time.time()
            print(f"Time taken to run the code was {end - start} seconds")

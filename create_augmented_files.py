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

from skimage.transform import resize
from scipy.ndimage import rotate


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
        im_zarr = zarr.open(f'{os.path.normpath(os.path.join(folder_path, filename))}', mode="r")
        im_shape = (im_zarr.shape[2], im_zarr.shape[0], im_zarr.shape[1])
    return im_shape


def get_chunk_bboxes(folder_path, filename, data_shape, input_is_zarr):
    im_shape = get_image_shape(folder_path, filename, input_is_zarr)

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
        chunk = im[bbox[0] - z_min:bbox[3] - z_min, bbox[1] - y_min:bbox[4] - y_min, bbox[2] - x_min:bbox[5] - x_min]
        within_bounds = (chunk > min_val) & (chunk < max_val)
        occ_ratios[i] = np.mean(within_bounds)

    return index, im, occ_ratios  # Return index and processed image


def convert_tiff_to_zarr(dataset, folder_path, channel_pattern, filenames, out_folder, out_name, batch_size,
                         input_is_zarr, date, channel_num, timepoint_i, output_zarr_version, outer_data_shape,
                         data_shape=None):
    if not data_shape:
        data_shape = [batch_size, 128, 128, 128]

    bboxes = get_chunk_bboxes(folder_path, filenames[0], data_shape, input_is_zarr)
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


def crop_center_cube(arr: np.ndarray, cube_size):
    z, y, x = arr.shape
    cz, cy, cx = cube_size

    start_z = (z - cz) // 2
    start_y = (y - cy) // 2
    start_x = (x - cx) // 2

    return arr[start_z:start_z + cz, start_y:start_y + cy, start_x:start_x + cx]


def augment_chunk(chunk, pixel_size, angle, orig_cube_size, crop_size):
    # Rotate in XY around Z
    chunk = rotate(chunk, angle=angle[1], axes=(1, 2), reshape=False, order=0)

    # Rotate in ZY around X
    chunk = rotate(chunk, angle=angle[0], axes=(0, 1), reshape=False, order=0)

    chunk = crop_center_cube(chunk, crop_size)

    return resize(chunk, (orig_cube_size, orig_cube_size, orig_cube_size), order=0, preserve_range=True,
                  anti_aliasing=False)


def augment_chunk_from_tile(filename, bbox, timepoint_num, channel_num, pixel_size, angle, orig_cube_size, crop_size):
    zarr_spec = {
        'driver': 'zarr3',
        'kvstore': {
            'driver': 'file',
            'path': filename
        }
    }
    zarr_file = ts.open(zarr_spec).result()

    chunk = zarr_file[timepoint_num, bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5], channel_num].read().result()

    return augment_chunk(chunk, pixel_size, angle, orig_cube_size, crop_size)


def get_chunk_bounding_box(i, chunk_size, num_chunks):
    #num_x, num_y, num_z = num_chunks
    #cx, cy, cz = chunk_size

    x = i % num_chunks[2]
    y = (i // num_chunks[2]) % num_chunks[1]
    z = (i // (num_chunks[2] * num_chunks[1])) % num_chunks[0]

    start_x = x * chunk_size[2]
    start_y = y * chunk_size[1]
    start_z = z * chunk_size[0]

    end_x = start_x + chunk_size[2]
    end_y = start_y + chunk_size[1]
    end_z = start_z + chunk_size[0]

    return [start_z, start_y, start_x, end_z, end_y, end_x]


if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--folder-paths', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="Paths to the folder containing the zarr tiles")
    ap.add_argument('--tile-filenames', type=lambda s: list(map(str, s.split(','))), required=True,
                    help="List of tile filenames that will be processed")
    ap.add_argument('--batch-start-number', type=int, required=True,
                    help="Number to start the batch of files at")
    ap.add_argument('--batch-size', type=int, required=True,
                    help="Number of chunks to process")
    ap.add_argument('--timepoint-num', type=int, required=True,
                    help="Number of the timepoint to process")
    ap.add_argument('--channel-num', type=int, required=True,
                    help="Number of the channel to process")
    ap.add_argument('--num-timepoints-per-image', type=int, default=16,
                    help="Number of timepoints in a training image")
    args = ap.parse_args()
    folder_paths = args.folder_paths
    tile_filenames = args.tile_filenames
    batch_start_number = args.batch_start_number
    batch_size = args.batch_size
    timepoint_num = args.timepoint_num
    channel_num = args.channel_num
    num_timepoints_per_image = args.num_timepoints_per_image

    batch_end_number = batch_start_number + batch_size

    for folder_path in folder_paths:
        with open(os.path.join(folder_path, 'metadata.json')) as f:
            metadata = json.load(f)
        start = time.time()

        for tile_name in tile_filenames:
            tile = metadata['training_images'][tile_name]
            orig_filename = os.path.join(metadata['input_folder'], tile_name)
            filename = os.path.join(metadata['server_folder'], metadata['output_folder'], tile_name)
            zarr_spec = {
                'driver': 'zarr3',
                'kvstore': {
                    'driver': 'file',
                    'path': filename
                }
            }
            zarr_file = ts.open(zarr_spec).result()
            for timepoint in range(timepoint_num, timepoint_num+num_timepoints_per_image):
                for i in range(batch_start_number, batch_end_number):
                    chunk = tile['chunk_names'][str(i)]
                    augmented_chunk = augment_chunk_from_tile(orig_filename, chunk['bbox'], timepoint, channel_num,
                                                              metadata['augmentation_info']['pixel_size'],
                                                              metadata['augmentation_info']['angle'], metadata['cube_size'],
                                                              metadata['augmentation_info']['new_size'])
                    bbox = get_chunk_bounding_box(i, [metadata['cube_size'], metadata['cube_size'], metadata['cube_size']],
                                                  [metadata['augmentation_info']['num_z_chunks'],
                                                   metadata['augmentation_info']['num_y_chunks'],
                                                   metadata['augmentation_info']['num_x_chunks']])

                    zarr_file[timepoint, bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5], channel_num] = augmented_chunk

        end = time.time()
        print(f"Time taken to run the code was {end - start} seconds")
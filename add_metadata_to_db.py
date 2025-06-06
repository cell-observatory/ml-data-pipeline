import argparse
import copy
import json
import re

from supabase import create_client


def get_first_key(d):
    for key in d:
        return key


def add_metadata_to_db(metadata_file, url, key):
    # Supabase client
    supabase = create_client(url, key)
    with open(metadata_file) as f:
        metadata = json.load(f)
    training_images = metadata.pop('training_images')

    bbox = training_images[get_first_key(training_images)]['bbox']

    prepared_entry = {
        'software_version': metadata.pop('software_version'),
        'output_folder': metadata.pop('output_folder'),
        'elapsed_sec': metadata.pop('elapsed_sec'),
        'cube_size': metadata.pop('cube_size'),
        'server_folder': metadata.pop('server_folder'),
        'time_size': metadata.pop('time_size'),
        'z_start': bbox[0],
        'y_start': bbox[1],
        'x_start': bbox[2],
        'z_end': bbox[3],
        'y_end': bbox[4],
        'x_end': bbox[5],
        'channel_size': metadata.pop('channel_size'),
        'metadata_json': metadata  # Rest of the data in logs will go to metadata
    }

    # Insert to prepared table
    response = supabase.table('prepared').insert(prepared_entry).execute()

    prepared_id = response.data[0]['id']
    prepared_cubes_entry_list = []
    delimiters = r"[./]"
    for training_image, training_image_dict in training_images.items():
        prepared_cubes_entry = {'prepared_id': prepared_id,
                                'tile_name': training_image}
        for chunk_name, chunk_metadata in training_image_dict['chunk_names'].items():
            curr_chunk_list = re.split(delimiters, chunk_name)
            curr_chunk_list.remove('c')
            prepared_cubes_entry_copy = copy.deepcopy(prepared_cubes_entry)
            prepared_cubes_entry_copy['chunk'] = curr_chunk_list[0]
            prepared_cubes_entry_copy['time'] = curr_chunk_list[1]
            bbox = chunk_metadata.pop('bbox')
            prepared_cubes_entry_copy['z_start'] = bbox[0]
            prepared_cubes_entry_copy['y_start'] = bbox[1]
            prepared_cubes_entry_copy['x_start'] = bbox[2]
            prepared_cubes_entry_copy['channel'] = curr_chunk_list[5]
            prepared_cubes_entry_copy['occupancy_ratio'] = chunk_metadata.pop('occ_ratio')
            if chunk_metadata:
                prepared_cubes_entry_copy['metadata_json'] = chunk_metadata
            prepared_cubes_entry_list.append(prepared_cubes_entry_copy)
    response = supabase.table('prepared_cubes').insert(prepared_cubes_entry_list).execute()
    return response


if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--metadata-file', type=str, required=True,
                    help="Full path to the metadata file")
    ap.add_argument('--url', type=str, required=True,
                    help="Database URL")
    ap.add_argument('--key', type=str, required=True,
                    help="Database Key")
    args = ap.parse_args()
    metadata_file = args.metadata_file
    url = args.url
    key = args.key

    response = add_metadata_to_db(metadata_file, url, key)

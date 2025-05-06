import argparse
import copy
import json
import re

from supabase import create_client, Client


def add_metadata_to_db(metadata_file, url, key):
    # Supabase client
    supabase = create_client(url, key)
    with open(metadata_file) as f:
        metadata = json.load(f)
    training_images = metadata.pop('training_images')
    channel_patterns_dict = {}
    for i, channel_pattern in enumerate(metadata['channelPatterns']):
        channel_patterns_dict[i] = channel_pattern

    prepared_entry = {
        'software_version': metadata.pop('software_version'),
        'output_folder': metadata.pop('output_folder'),
        'elapsed_sec': metadata.pop('elapsed_sec'),
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
            prepared_cubes_entry_copy['chunk_name'] = chunk_name
            prepared_cubes_entry_copy['chunk'] = curr_chunk_list[0]
            prepared_cubes_entry_copy['time'] = curr_chunk_list[1]
            prepared_cubes_entry_copy['z'] = curr_chunk_list[2]
            prepared_cubes_entry_copy['y'] = curr_chunk_list[3]
            prepared_cubes_entry_copy['x'] = curr_chunk_list[4]
            prepared_cubes_entry_copy['channel'] = curr_chunk_list[5]
            prepared_cubes_entry_copy['occupancy_ratio'] = chunk_metadata.pop('occ_ratio')
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

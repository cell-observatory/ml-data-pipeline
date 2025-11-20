import argparse
import copy
import json
import math
import os
import pandas as pd

from supabase import create_client
from supabase.lib.client_options import ClientOptions, SyncClientOptions


def add_synthetic_metadata_to_db(metadata_file, url, key):
    # Supabase client
    supabase = create_client(url, key,
                             options=SyncClientOptions(postgrest_client_timeout=600, storage_client_timeout=600,
                                                       schema="public"))
    # Get the directory that contains prepared.csv
    folder = os.path.dirname(metadata_file)

    # Build paths to the other two CSVs
    tiles_path = os.path.join(folder, "prepared_tiles.csv")
    cubes_path = os.path.join(folder, "prepared_cubes.csv")

    # Read all three
    df_prepared = pd.read_csv(metadata_file)
    df_prepared_tiles = pd.read_csv(tiles_path)
    df_prepared_cubes = pd.read_csv(cubes_path)

    for prepared_row in df_prepared.itertuples(index=False):

        prepared_entry = {
            'software_version': prepared_row.software_version,
            'output_folder': prepared_row.output_folder,
            'elapsed_sec': 0,
            'cube_size': prepared_row.cube_size,
            'server_folder': prepared_row.server_folder,
            'time_size': prepared_row.time_size,
            'data_location': prepared_row.data_location,
            'z_start': prepared_row.z_start,
            'y_start': prepared_row.y_start,
            'x_start': prepared_row.x_start,
            'z_end': prepared_row.z_end,
            'y_end': prepared_row.y_end,
            'x_end': prepared_row.x_end,
            'channel_size': prepared_row.channel_size,
            'is_synthetic': prepared_row.is_synthetic
        }

        prepared_id = None
        response = None
        try:
            # Insert to prepared table
            response = supabase.table('prepared').insert(prepared_entry).execute()

            prepared_id = response.data[0]['id']

            prepared_tiles_entry_list = []
            tile_rows = df_prepared_tiles[df_prepared_tiles["prepared_id"] == prepared_row.id]
            for tile_row in tile_rows.itertuples(index=False):
                prepared_tiles_entry = {'prepared_id': prepared_id,
                                        'tile_name': tile_row.tile_name}
                prepared_tiles_entry_list.append(prepared_tiles_entry)
            response = supabase.table('prepared_tiles').insert(prepared_tiles_entry_list).execute()

            prepared_cubes_entry_list = []
            cube_rows = df_prepared_cubes[df_prepared_cubes["prepared_id"] == prepared_row.id]
            for cube_row in cube_rows.itertuples(index=False):
                prepared_cubes_entry = {'prepared_id': prepared_id,
                                        'tile_name': cube_row.tile_name}
                prepared_cubes_entry_copy = copy.deepcopy(prepared_cubes_entry)
                prepared_cubes_entry_copy['chunk'] = cube_row.chunk
                prepared_cubes_entry_copy['time'] = cube_row.time
                prepared_cubes_entry_copy['z_start'] = cube_row.z_start
                prepared_cubes_entry_copy['y_start'] = cube_row.y_start
                prepared_cubes_entry_copy['x_start'] = cube_row.x_start
                prepared_cubes_entry_copy['channel'] = cube_row.channel
                occ_ratio = cube_row.occupancy_ratio
                if not math.isnan(occ_ratio):
                    prepared_cubes_entry_copy['occupancy_ratio'] = cube_row.occupancy_ratio
                prepared_cubes_entry_copy['metadata_json'] = json.loads(cube_row.metadata_json)
                prepared_cubes_entry_copy['channel_target'] = cube_row.channel_target
                prepared_cubes_entry_list.append(prepared_cubes_entry_copy)

            # Insert 10000 cube entries at a time if possible
            insert_batch_size = 10000
            num_cube_entries = len(prepared_cubes_entry_list)
            insert_batch_size = min(insert_batch_size, num_cube_entries)
            for i in range(0, num_cube_entries, insert_batch_size):
                response = supabase.table('prepared_cubes').insert(
                    prepared_cubes_entry_list[i:i + insert_batch_size]).execute()
        except:
            if prepared_id is not None:
                supabase.table('prepared').delete().eq('id', prepared_id).execute()
            raise Exception(f'Insertion failed. Response: {response}')
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

    response = add_synthetic_metadata_to_db(metadata_file, url, key)

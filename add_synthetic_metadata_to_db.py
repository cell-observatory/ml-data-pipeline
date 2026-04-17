import argparse
import json
import math
import os
import pandas as pd

from supabase import create_client
from supabase.lib.client_options import SyncClientOptions


CLUSTER_CHOICES = ("prfs", "aws", "oak", "abc", "nersc", "gcp")


def add_synthetic_metadata_to_db(metadata_file, url, key, cluster=None):
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
    df_prepared_cubes = pd.read_csv(cubes_path)!

    for prepared_row in df_prepared.itertuples(index=False):

        prepared_entry = {
            'software_version': prepared_row.software_version,
            'output_folder': prepared_row.output_folder,
            'elapsed_sec': 0,
            'cube_size': prepared_row.cube_size,
            'server_folder': prepared_row.server_folder,
            'time_size': int(prepared_row.time_size),
            'data_location': prepared_row.data_location,
            'channel_size': prepared_row.channel_size,
            'is_synthetic': prepared_row.is_synthetic,
            'is_available': True,
        }
        if cluster:
            prepared_entry[f'exists_{cluster}'] = True
        if hasattr(prepared_row, 'channel_mapping') and pd.notna(prepared_row.channel_mapping):
            prepared_entry['channel_mapping'] = json.loads(prepared_row.channel_mapping) \
                if isinstance(prepared_row.channel_mapping, str) else prepared_row.channel_mapping
        if hasattr(prepared_row, 'raw_roi_acquisition_id') and pd.notna(prepared_row.raw_roi_acquisition_id):
            prepared_entry['raw_roi_acquisition_id'] = prepared_row.raw_roi_acquisition_id

        prepared_id = None
        response = None
        try:
            # Insert to prepared table
            response = supabase.table('prepared').insert(prepared_entry).execute()
            prepared_id = response.data[0]['id']

            prepared_tiles_entry_list = []
            tile_rows = df_prepared_tiles[df_prepared_tiles["prepared_id"] == prepared_row.id]
            for tile_row in tile_rows.itertuples(index=False):
                tile_entry = {
                    'prepared_id': prepared_id,
                    'tile_name': tile_row.tile_name,
                    'is_test_split': getattr(tile_row, 'is_test_split', False),
                    'channel_size': getattr(tile_row, 'channel_size', None),
                }
                if hasattr(tile_row, 'time_size') and pd.notna(tile_row.time_size):
                    tile_entry['time_size'] = int(tile_row.time_size)
                if hasattr(tile_row, 'n_timepoints') and pd.notna(tile_row.n_timepoints):
                    tile_entry['n_timepoints'] = int(tile_row.n_timepoints)
                for dim in ['n_z', 'n_y', 'n_x', 'raw_n_z', 'raw_n_y', 'raw_n_x']:
                    if hasattr(tile_row, dim) and pd.notna(getattr(tile_row, dim)):
                        tile_entry[dim] = int(getattr(tile_row, dim))
                for gb_col in ['raw_size_gb', 'processed_size_gb']:
                    if hasattr(tile_row, gb_col) and pd.notna(getattr(tile_row, gb_col)):
                        tile_entry[gb_col] = float(getattr(tile_row, gb_col))
                prepared_tiles_entry_list.append(tile_entry)
            response = supabase.table('prepared_tiles').insert(prepared_tiles_entry_list).execute()

            prepared_cubes_entry_list = []
            cube_rows = df_prepared_cubes[df_prepared_cubes["prepared_id"] == prepared_row.id]
            for cube_row in cube_rows.itertuples(index=False):
                cube_entry = {
                    'prepared_id': prepared_id,
                    'tile_name': cube_row.tile_name,
                    'chunk': cube_row.chunk,
                    'time': cube_row.time,
                    'z_start': cube_row.z_start,
                    'y_start': cube_row.y_start,
                    'x_start': cube_row.x_start,
                    'channel': cube_row.channel,
                }
                occ_ratio = cube_row.occupancy_ratio
                if not math.isnan(occ_ratio):
                    cube_entry['occupancy_ratio'] = cube_row.occupancy_ratio
                for cdf_col in ['cdf_80', 'cdf_90', 'cdf_95', 'cdf_99']:
                    if hasattr(cube_row, cdf_col) and pd.notna(getattr(cube_row, cdf_col)):
                        cube_entry[cdf_col] = int(getattr(cube_row, cdf_col))
                prepared_cubes_entry_list.append(cube_entry)

            insert_batch_size = 10000
            num_cube_entries = len(prepared_cubes_entry_list)
            insert_batch_size = min(insert_batch_size, num_cube_entries)
            for i in range(0, num_cube_entries, insert_batch_size):
                response = supabase.table('prepared_cubes').insert(
                    prepared_cubes_entry_list[i:i + insert_batch_size]).execute()

            supabase.rpc('refresh_prepared_cache_artifacts', {'p_prepared_id': prepared_id}).execute()
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
    ap.add_argument('--cluster', type=str, default=None,
                    choices=CLUSTER_CHOICES,
                    help="Storage location where processed data resides; "
                         "sets the matching exists_<cluster> flag on each new "
                         "prepared row.")
    args = ap.parse_args()

    response = add_synthetic_metadata_to_db(
        args.metadata_file, args.url, args.key, cluster=args.cluster,
    )
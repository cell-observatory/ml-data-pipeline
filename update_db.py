import argparse
import logging
import json
from supabase import create_client, Client

def add_entry(supabase: Client, log_file: str):
    with open(log_file) as f:
        logs = json.load(f)

    if len(logs) < 4:
        raise Exception("Log file has missing entries.")

    if 'input_folder' in logs:
        input_folder = logs['input_folder']
    else:
        raise Exception("Input folder is a required field for the log file.")

    entry = {
        'software_version': logs.pop('software_version'),
        'output_folder': logs.pop('output_folder'),
        'elapsed_sec': logs.pop('elapsed_sec'),
        'metadata_json': logs # Rest of the data in logs will go to metadata
    }

    # Find acquisition id by matching the data_folder field in the acquisition table
    response = supabase.table('acq').select('id').eq('data_folder', input_folder).execute()
    if len(response.data) == 0:
        logging.info("Unique acquisition id is not found. Make sure to update the acquisition table with the correct information.")
    elif len(response.data) > 1:
        logging.info(f"Found {len(response.data)} unique acquisition ids. First one will be used.")
        entry['acquisition_id'] = response.data[0]['id']
    else:
        entry['acquisition_id'] = response.data[0]['id']

    # Insert to prepared table
    response = supabase.table('prepared').insert(entry).execute()

    return response

if __name__ == '__main__':
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', type=str, required=True,
                    help="Path to the folder to output the job logs to")
    ap.add_argument('--url', type=str, required=True,
                    help="Database URL")
    ap.add_argument('--key', type=str, required=True,
                    help="Database key")
    args = ap.parse_args()
    url = args.url
    key = args.key
    log_dir = args.log_dir

    # Supabase client
    supabase = create_client(url, key)

    # Add response to DB
    response = add_entry(supabase, log_dir)

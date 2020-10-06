import os

def sync_open_dataset(dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets/hdsm_small" "{}"'.format(dataset_dir)
    print(command)
    os.system(command)
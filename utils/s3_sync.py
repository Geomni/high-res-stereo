import os

def sync_dataset(dataset_dir):
    target_dir = os.path.join(dataset_dir, 'hdsm')
    os.makedirs(target_dir, exist_ok=True)
    command = 'aws s3 sync "s3://autogpe-datasets//hdsm_small" "{}" --dryrun'.format(target_dir)
    os.system(command)
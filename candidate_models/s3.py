import logging
import os

import boto3

_logger = logging.getLogger(__name__)

default_bucket = 'brain-score-models'
default_region = 'us-east-1'


def download_folder(folder_key, target_directory, bucket=default_bucket, region=default_region):
    if not folder_key.endswith('/'):
        folder_key = folder_key + '/'
    s3 = boto3.resource('s3', region_name=region)
    bucket = s3.Bucket(bucket)
    bucket_contents = list(bucket.objects.all())
    files = [obj.key for obj in bucket_contents if obj.key.startswith(folder_key)]
    _logger.debug(f"Found {len(files)} files")
    for file in files:
        # get part of file after given folder_key
        filename = file[len(folder_key):]
        target_path = os.path.join(target_directory, filename)
        temp_path = target_path + '.filepart'
        bucket.download_file(file, temp_path)
        os.rename(temp_path, target_path)

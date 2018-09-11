import os
import tempfile

import logging

from candidate_models import s3


class TestDownloadFolder:
    def test_mobilenet_weights(self):
        logging.getLogger('botocore').setLevel(logging.WARNING)
        logging.getLogger('s3transfer').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)

        model = 'mobilenet_v1_0.25_128'
        with tempfile.TemporaryDirectory() as target_path:
            s3.download_folder(f"slim/{model}", target_path)
            downloaded_files = os.listdir(target_path)
            expected_suffixes = ['.ckpt.data-00000-of-00001', '.ckpt.index', '.ckpt.meta', '.tflite',
                                 '_eval.pbtxt', '_frozen.pb', '_info.txt']
            assert downloaded_files == [model + suffix for suffix in expected_suffixes]

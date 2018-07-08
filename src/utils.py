import hashlib
import logging
import os
import tarfile
from urllib.parse import urlsplit
from urllib.request import urlretrieve

from tqdm import tqdm


def sha256_checksum(file_name, block_size=65536):
    if not os.path.isfile(file_name):
        return None
    sha256 = hashlib.sha256()
    with open(file_name, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            sha256.update(block)
    return sha256.hexdigest()


class DownloadProgressBar(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def download_dataset():
    base_dir = os.getcwd()
    data_url = 'https://pubdata.endgame.com/ember/ember_dataset.tar.bz2'
    data_checksum = 'a5603de2f34f02ab6e21df7a0f97ec4ac84ddc65caee33fb610093dd6f9e1df9'
    data_file = os.path.join(base_dir, os.path.split(urlsplit(data_url).path)[-1])
    data_dir = os.path.join(base_dir, 'ember')

    if sha256_checksum(data_file) != data_checksum:
        logging.info('Downloading EMBER dataset from {}'.format(urlsplit(data_url).hostname))
        with DownloadProgressBar(unit='B', unit_scale=True, ncols=96, miniters=1, desc='EMBER Dataset') as bar:
            urlretrieve(data_url, data_file, bar.hook)
    else:
        logging.info('The EMBER dataset is existed at {}'.format(data_file))

    if not os.path.exists(data_dir):
        logging.info('Extracting the dataset')
        with tarfile.open(data_file) as tar:
            tar.extractall(base_dir)
    else:
        logging.info('Data directory is existed at {}'.format(data_dir))

    logging.info('The dataset is ready')

import json
import logging
import multiprocessing as mp
import os
import sys

import numpy as np
from tqdm import tqdm

from utils import sha256_checksum
from features import FeatureExtractor


def raw_feature_iterator(file_paths):
    for path in file_paths:
        with open(path, "r") as f:
            for line in f:
                yield line


def vectorize_data(arg):
    row, raw_data, x_path, y_path, n_rows = arg
    extractor = FeatureExtractor()
    dim = FeatureExtractor.dim
    raw_features = json.loads(raw_data)
    y = np.memmap(y_path, dtype=np.float32, mode="r+", shape=n_rows)
    y[row] = raw_features["label"]
    feature_vector = extractor.process_raw_features(raw_features)
    x = np.memmap(x_path, dtype=np.float32, mode="r+", shape=(n_rows, dim))
    x[row] = feature_vector


def vectorize_subset(subset):
    # Checking dataset
    data_dir = os.path.join(os.getcwd(), 'ember')
    if subset == 'train':
        paths = [os.path.join(data_dir, "train_features_{}.jsonl".format(i)) for i in range(6)]
        n_rows = 900000
    elif subset == 'test':
        paths = [os.path.join(data_dir, "test_features.jsonl"), ]
        n_rows = 200000
    else:
        logging.error('subset must be "train" or "test"')
        sys.exit(1)
    for p in paths:
        if not os.path.exists(p):
            logging.info('File not found: {}'.format(p))
            sys.exit(1)
    X_path = os.path.join(data_dir, "X_{}.dat".format(subset))
    y_path = os.path.join(data_dir, "y_{}.dat".format(subset))

    if os.path.exists(X_path + '.shd256') and os.path.exists(y_path + '.shd256'):
        with open(X_path + '.shd256', 'r') as f:
            X_checksum = f.read()
        with open(y_path + '.shd256', 'r') as f:
            y_checksum = f.read()
        if X_checksum == sha256_checksum(X_path) and y_checksum == sha256_checksum(y_path):
            logging.info('"{}" subset is vectorized'.format(subset))
            return

    # Allocate storage space
    dim = FeatureExtractor.dim
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(n_rows, dim))
    y = np.memmap(y_path, dtype=np.float32, mode="w+", shape=n_rows)
    del X, y

    logging.info('Vectorzing samples in "{}" subset'.format(subset))
    pool = mp.Pool()
    arg_iterator = ((row, raw_data, X_path, y_path, n_rows)
                    for row, raw_data in enumerate(raw_feature_iterator(paths)))
    for _ in tqdm(pool.imap_unordered(vectorize_data, arg_iterator),
                  unit='row', unit_scale=True, ncols=96, miniters=1, total=n_rows):
        pass

    X_checksum = sha256_checksum(X_path)
    with open(X_path + '.shd256', 'w') as f:
        f.write(X_checksum)
    y_checksum = sha256_checksum(y_path)
    with open(y_path + '.shd256', 'w') as f:
        f.write(y_checksum)


def save_numpy_file(subset):
    data_dir = os.path.join(os.getcwd(), 'ember')
    if subset == 'train':
        n_rows = 900000
    elif subset == 'test':
        n_rows = 200000
    else:
        logging.error('subset must be "train" or "test"')
        sys.exit(1)

    X_npy = os.path.join(data_dir, "X_{}.npy".format(subset))
    y_npy = os.path.join(data_dir, "y_{}.npy".format(subset))

    if os.path.exists(X_npy + '.shd256') and os.path.exists(y_npy + '.shd256'):
        with open(X_npy + '.shd256', 'r') as f:
            X_checksum = f.read()
        with open(y_npy + '.shd256', 'r') as f:
            y_checksum = f.read()
        if X_checksum == sha256_checksum(X_npy) and y_checksum == sha256_checksum(y_npy):
            logging.info('Numpy files of "{}" subset is existed!'.format(subset))
            return

    logging.info('Saving numpy files for labeled samples in "{}" subset'.format(subset))
    dim = FeatureExtractor.dim
    X_dat = os.path.join(data_dir, "X_{}.dat".format(subset))
    y_dat = os.path.join(data_dir, "y_{}.dat".format(subset))
    X = np.memmap(X_dat, dtype=np.float32, mode="r", shape=(n_rows, dim))
    y = np.memmap(y_dat, dtype=np.float32, mode="r", shape=n_rows)
    labeled_rows = (y != -1)
    np.save(X_npy, X[labeled_rows])
    np.save(y_npy, y[labeled_rows])

    X_checksum = sha256_checksum(X_npy)
    with open(X_npy + '.shd256', 'w') as f:
        f.write(X_checksum)
    y_checksum = sha256_checksum(y_npy)
    with open(y_npy + '.shd256', 'w') as f:
        f.write(y_checksum)


def vectorize_dataset():
    dim = FeatureExtractor.dim
    logging.info('Dimension: {}'.format(dim))
    vectorize_subset('train')
    save_numpy_file('train')
    vectorize_subset('test')
    save_numpy_file('test')
    logging.info('Dataset is vectorized')

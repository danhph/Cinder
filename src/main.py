import argparse
import os

from logger import init_logger
from utils import download_dataset
from vectorize import vectorize_dataset
from model import train_model
from model import evaluate_model


def main():
    # download_dataset()
    # vectorize_dataset()
    # train_model()
    evaluate_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cwd', type=str, metavar='WORKING_DIR', help='Set Cinder working directory')
    options = parser.parse_args()
    if options.cwd is not None and options.cwd != os.getcwd():
        os.chdir(options.cwd)
    init_logger()
    main()

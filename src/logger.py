import logging
import os


def init_logger():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s", "%Y-%m-%d %H:%M:%S")
    # Write logs to file
    file_handler = logging.FileHandler(os.path.join(os.getcwd(), 'cinder.log'))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)
    # Write logs to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

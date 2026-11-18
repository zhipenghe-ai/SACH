"""Log baseline training and evaluation runs."""

import logging
import os.path


def logger(fileName="log", log_dir="./logs"):
    run_logger = logging.getLogger("train")
    run_logger.setLevel(logging.INFO)
    log_format = "%(asctime)s - %(levelname)s - %(message)s"

    file_handler = logging.FileHandler(os.path.join(log_dir, str(fileName) + ".txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    run_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(log_format))
    run_logger.addHandler(stream_handler)
    return run_logger

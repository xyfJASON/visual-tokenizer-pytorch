import tqdm
import logging


def get_logger(
        name: str = 'exp',
        log_file: str = None,
        log_level: int = logging.INFO,
        file_mode: str = 'w',
        use_tqdm_handler: bool = False,
        is_main_process: bool = True,
):
    logger = logging.getLogger(name)
    # Check if the logger exists
    if logger.hasHandlers():
        return logger
    # Add a stream handler
    if not use_tqdm_handler:
        stream_handler = logging.StreamHandler()
    else:
        stream_handler = TqdmLoggingHandler()
    handlers = [stream_handler]
    # Add a file handler for main process
    if is_main_process and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)
    # Set format & level for all handlers
    # Note that levels of non-master processes are always 'ERROR'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_level = log_level if is_main_process else logging.ERROR
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level: int = logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:  # noqa
            self.handleError(record)

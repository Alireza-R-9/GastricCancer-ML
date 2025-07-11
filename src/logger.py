import logging

def setup_logger(name='classifier_logger', log_file='training.log'):
    """
    Configure and return a logger for recording training process logs to a file.

    Parameters:
    - name: str, optional (default='classifier_logger')
        Name of the logger.
    - log_file: str, optional (default='training.log')
        File path where logs will be saved.

    Returns:
    - logger: logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create file handler which logs messages to a specified file
    fh = logging.FileHandler(log_file)

    # Define log message format: timestamp - log level - message
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(fh)

    return logger

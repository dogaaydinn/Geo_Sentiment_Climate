import logging
import colorlog
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_file: str, log_level=logging.INFO) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False

    if not logger.handlers:
        # File handler with rotation
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        fh.setLevel(log_level)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

        # Console handler with color
        console_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(console_formatter)
        logger.addHandler(ch)

    return logger
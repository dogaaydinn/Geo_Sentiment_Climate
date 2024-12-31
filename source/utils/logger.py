import logging
from logging.handlers import RotatingFileHandler

def setup_logger(name="root", log_file="logs/app.log", log_level="INFO"):
    """
    Logger oluşturur ve yapılandırır. Hem dosyaya hem konsola log yazar.

    Args:
        name (str): Logger adı.
        log_file (str): Log dosyası yolu.
        log_level (str): Log seviyesi (örn: "INFO", "DEBUG", "ERROR").
    """
    # Logger'ı oluştur
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=2)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Handlers'ı ekle (Duplicate eklemeyi önler)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

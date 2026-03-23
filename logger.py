import logging
import sys

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured standard logger.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
    return logger

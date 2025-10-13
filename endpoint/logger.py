import logging
import sys
class Logger:
    @staticmethod
    def get_logger():
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        # Remove old handlers to avoid duplicate logs
        for h in list(logger.handlers):
            logger.removeHandler(h)

        # Console handler
        stream_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

        # File handler with detailed format
        file_formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d | %(levelname)-8s | "
            "%(name)s:%(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler = logging.FileHandler("logs.log", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        return logger

logger = Logger.get_logger()
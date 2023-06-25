"""logging模块的封装
"""
import logging
from operator import truediv
from tempfile import TemporaryDirectory
from typing import Optional


logger_initialized = {}

def get_logger(name: str, 
                log_file: Optional[str] = None,
                log_level: int = logging.INFO,
                file_mode: str = 'w' ):
    """返回一个设置好的logger

    Args:
        name (str): 该logger的名字
        log_file (Optional[str], optional): 把log写入文件的名字
        log_level (int, optional): log的级别，默认为info
        file_mode (str, optional): 写入log文件的模式，默认为write

    Returns:
       返回设置好的logger
    """
    logger = logging.get_logger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger
    
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logger.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:

        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name) - %(level_name)s - %(message)s'
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    logger.setLevel(log_level)
    logger_initialized[name] = True

    return logger
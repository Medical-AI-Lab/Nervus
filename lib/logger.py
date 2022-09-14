#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import logging


class BaseLogger:
    """
    Class for defining logger.
    """
    _unexecuted_configure = True

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Set logger.

        Args:
            name (str): If needed, potentially hierarchical name is desired, eg. lib.net, lib.dataloader, etc.
                        For the details, see https://docs.python.org/3/library/logging.html?highlight=logging#module-logging.
        Returns:
            logging.Logger: logger
        """
        if cls._unexecuted_configure:
            cls._init_logger()

        return logging.getLogger('nervus.{}'.format(name))

    @classmethod
    def set_level(cls, level: int) -> None:
        """
        Set logging level.

        Args:
            level (int): logging level
        """
        _root_logger = logging.getLogger('nervus')
        _root_logger.setLevel(level)

    @classmethod
    def _init_logger(cls) -> None:
        """
        Configure logger.
        """
        _root_logger = logging.getLogger('nervus')
        _root_logger.setLevel(logging.INFO)

        # file handler
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        log_path = Path(log_dir, 'log.log')
        fh = logging.FileHandler(log_path)
        _root_logger.addHandler(fh)

        # uppper warining
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        format = logging.Formatter('%(levelname)-8s %(message)s')
        ch.setFormatter(format)
        ch.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
        _root_logger.addHandler(ch)

        # lower warning
        ch_info = logging.StreamHandler()
        ch_info.setLevel(logging.DEBUG)
        ch_info.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
        _root_logger.addHandler(ch_info)

        cls._unexecuted_configure = False


class Logger:
    """
    Class to handle logger as global.

    As default, set logger which does nothing.
    """
    logger = logging.getLogger('null')
    logger.addHandler(logging.NullHandler())


def set_logger() -> None:
    """
    Set logger by orverwriting the default Logger.logger.
    """
    Logger.logger = BaseLogger.get_logger('logs')

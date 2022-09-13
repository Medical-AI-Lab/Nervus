#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import logging


class BaseLogger:
    _unexecuted_configure = True

    @classmethod
    def get_logger(cls, filename):
        if cls._unexecuted_configure:
            cls._init_logger(filename)

        return logging.getLogger('nervus.{}'.format(filename))

    @classmethod
    def set_level(cls, level):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(level)

    @classmethod
    def _init_logger(cls, filename):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(logging.INFO)

        # file handler
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        log_path = Path(log_dir, filename).with_suffix('.log')
        fh = logging.FileHandler(log_path)
        _nervus_root_logger.addHandler(fh)

        # uppper warining
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        format = logging.Formatter('%(levelname)-8s %(message)s')
        ch.setFormatter(format)
        ch.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
        _nervus_root_logger.addHandler(ch)

        # lower warning
        ch_info = logging.StreamHandler()
        ch_info.setLevel(logging.DEBUG)
        ch_info.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
        _nervus_root_logger.addHandler(ch_info)

        cls._unexecuted_configure = False


class Logger:
    """
    Class to handle logger as global

    As default, set logger which does nothing.
    """
    logger = logging.getLogger('Null')
    logger.addHandler(logging.NullHandler())


def set_logger():
    Logger.logger = BaseLogger.get_logger('log')

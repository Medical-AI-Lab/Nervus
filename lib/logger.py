#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging


class Logger:
    _unexecuted_configure = True

    @classmethod
    def get_logger(cls, filename):
        if cls._unexecuted_configure:
            cls._init_logger()

        return logging.getLogger('nervus.{}'.format(filename))

    @classmethod
    def set_level(cls, level):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(level)

    @classmethod
    def _init_logger(cls):
        _nervus_root_logger = logging.getLogger('nervus')
        _nervus_root_logger.setLevel(logging.INFO)

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


def get_logger(filename):
    logger = Logger.get_logger(filename)
    return logger

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
    def _init_logger(cls) -> None:
        """
        Configure logger.
        """
        _root_logger = logging.getLogger('nervus')
        _root_logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        log_dir = Path('logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir, 'log.log')

        # file handler
        ## upper warning
        fh_err = logging.FileHandler(log_path)
        fh_err.setLevel(logging.WARNING)
        fh_err.setFormatter(formatter)
        fh_err.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
        _root_logger.addHandler(fh_err)

        ## lower warning
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
        _root_logger.addHandler(fh)

        # stream handler
        ## upper warning
        ch_err = logging.StreamHandler()
        ch_err.setLevel(logging.WARNING)
        ch_err.setFormatter(formatter)
        ch_err.addFilter(lambda log_record: log_record.levelno >= logging.WARNING)
        _root_logger.addHandler(ch_err)

        ## lower warning
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.addFilter(lambda log_record: log_record.levelno < logging.WARNING)
        _root_logger.addHandler(ch)

        cls._unexecuted_configure = False

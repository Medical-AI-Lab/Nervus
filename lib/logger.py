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

        # file handler
        log_dir = Path('./logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = Path(log_dir, 'log.log')
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        _root_logger.addHandler(fh)

        # stream handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        _root_logger.addHandler(ch)

        cls._unexecuted_configure = False

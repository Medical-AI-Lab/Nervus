#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json
from lib import set_eval, BaseLogger
from typing import List


logger = BaseLogger.get_logger(__name__)


class EvalOptions:
    """
    Class for options.
    """
    def __init__(self) -> None:
        """
        Options for evaluation.
        """
        self.parser = argparse.ArgumentParser(description='Options for evaluation')
        self.parser.add_argument('--likelihood_dir', type=str, default=None, help='Directory of likekihoods (Default: None)')
        self.args = self.parser.parse_args()


    def _get_latest_likelihood_dir(self) -> str:
        """
        Return the latest directory of likelihood.

        Returns:
            str: Directory of likelihood
        """
        _likelihood_dirs = list(Path('results').glob('*/trials/*/likelihoods'))
        assert (_likelihood_dirs != []), 'No directory of likelihood.'
        _likelihood_dir = max(_likelihood_dirs, key=lambda likelihood_dir: Path(likelihood_dir).stat().st_mtime)
        return str(_likelihood_dir)

    def parse(self) -> None:
        """
        Parse options.
        """
        if self.args.likelihood_dir is None:
            _likelihood_dir = self._get_latest_likelihood_dir()
            setattr(self.args, 'likelihood_dir', _likelihood_dir)

        _datetime = Path(self.args.likelihood_dir).parents[0].name
        setattr(self.args, 'datetime',  _datetime)


def collect_likelihood(likelihood_dir: str) -> List[Path]:
    """
    Return list of likelihood paths.

    Args:
        datetime_dir (str): directory of datetime including likelihood

    Returns:
        List[Path]: list of likelihood paths
    """
    likelihood_paths = list(Path(likelihood_dir).glob('*.csv'))
    assert likelihood_paths != [], f"No likelihood in {likelihood_dir}."
    likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return likelihood_paths


def check_task(datetime_name: str) -> str:
    """
    Return task done on datetime_name

    Args:
        datetime_dir (str): directory of datetime at traning

    Returns:
        str: task
    """
    _parameter_path = list(Path('results').glob('*/trials/' + datetime_name + '/parameters.json'))[0]
    # If you specify the likelihood of an external dataset,
    # the project name is different from project name in training,
    # but the datetime_name is the same as the datatime in training, which is always one.
    with open(_parameter_path) as f:
        _parameters = json.load(f)
    task = _parameters['task']
    return task


def check_eval_options() -> EvalOptions:
    """
    Parse options for evaluation.

    Returns:
        EvalOptions: options
    """
    opt = EvalOptions()
    opt.parse()
    return opt


def main(opt):
    args = opt.args
    task = check_task(args.datetime)
    task_eval = set_eval(task)
    likelihood_paths = collect_likelihood(args.likelihood_dir)

    logger.info(f"Calculating metrics of {task} for {args.likelihood_dir}.\n")

    for likelihood_path in likelihood_paths:
        logger.info(likelihood_path.name)
        task_eval.make_metrics(likelihood_path)
        logger.info('\n')

    logger.info('Updated summary.')


if __name__ == '__main__':
    try:
        logger.info('\nEvaluation started.\n')

        opt = check_eval_options()
        main(opt)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nEvaluation finished.\n')


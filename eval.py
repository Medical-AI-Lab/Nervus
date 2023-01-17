#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import glob
import re
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
        self.parser.add_argument('--datetime_dir', type=str, default=None, help='Directory of datetime contains target likekihoods (Default: None)')
        self.args = self.parser.parse_args()

    def _get_latest_likelihood_dir(self) -> str:
        """
        Return the latest directory of likelihood.

        Returns:
            str: Directory of likelihood

        Note that:
        If directory of materials is link, Path('.').glob('**/likelihoods') cannot follow below materials.
        Therefore, use glob.glob.
        """
        _likelihood_dirs = glob.glob('**/results/*/sets/*/likelihoods', recursive=True)
        assert (_likelihood_dirs != []), 'No directory of likelihood.'
        _likelihood_dir = max(_likelihood_dirs, key=lambda likelihood_dir: Path(likelihood_dir).stat().st_mtime)
        return str(_likelihood_dir)

    def parse(self) -> None:
        """
        Parse options.
        """
        if self.args.datetime_dir is None:
            _likelihood_dir = self._get_latest_likelihood_dir()
            _datetime_dir = str(Path(_likelihood_dir).parents[0])
            setattr(self.args, 'datetime_dir',  _datetime_dir)


def check_task(likelihood_dir: str) -> str:
    """
    Return task.

    Args:
        likelihood_dir (str): Directory of likelihood

    Returns:
        str: task
    """
    _dataset_dir = re.findall('(.*)/results', likelihood_dir)[0]
    _datetime_dir = Path(likelihood_dir).name
    _parameter_path = list(Path(_dataset_dir).glob('results/*/sets/' + _datetime_dir + '/parameters.json'))[0]
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


def collect_likelihood(datetime_dir: str) -> List[Path]:
    """
    Return list of likelihood paths.

    Args:
        datetime_dir (str): path to directory of likelihoods

    Returns:
        List[Path]: list of likelihood paths
    """
    likelihood_paths = list(Path(datetime_dir, 'likelihoods').glob('*.csv'))
    assert likelihood_paths != [], f"No likelihood in {datetime_dir}."
    likelihood_paths.sort(key=lambda path: path.stat().st_mtime)
    return likelihood_paths


def main(opt):
    args = opt.args
    likelihood_paths = collect_likelihood(args.datetime_dir)
    task = check_task(args.datetime_dir)
    task_eval = set_eval(task)

    logger.info(f"Calculating metrics of {task} for {args.datetime_dir}.\n")

    for likelihood_path in likelihood_paths:
        logger.info(likelihood_path.name)
        task_eval.make_metrics(likelihood_path)
        logger.info('')

    logger.info('\nUpdated summary.')


if __name__ == '__main__':
    try:
        logger.info('\nEvaluation started.\n')

        opt = check_eval_options()
        main(opt)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('Evaluation done.\n')

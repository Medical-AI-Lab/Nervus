#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from lib import (
        check_test_options,
        set_params,
        create_model,
        set_logger
        )
from lib.component import (
        create_dataloader,
        set_likelihood
        )
from lib import Logger as logger
from typing import List


def _collect_weight(weight_dir: str) -> List[Path]:
    """
    Return list of weight paths.

    Args:
        weight_dir (st): path to directory of weights

    Returns:
        List[Path]: list of weight paths
    """
    weight_paths = list(Path(weight_dir).glob('*.pt'))
    assert weight_paths != [], f"No weight in {weight_dir}."
    weight_paths.sort(key=lambda path: path.stat().st_mtime)
    return weight_paths


def main(opt):
    params = set_params(opt.args)
    params.print_parameter()
    params.print_dataset_info()

    dataloaders = params.dataloaders
    model = create_model(params)
    likelihood = set_likelihood(params.task, params.num_outputs_for_label, params.save_datetime_dir)

    weight_paths = _collect_weight(params.weight_dir)
    for weight_path in weight_paths:
        logger.logger.info(f"Inference with {weight_path.name}.")

        # weight is reset by overwriting every time.
        model.load_weight(weight_path)
        model.eval()

        likelihood.init_likelihood()
        for split in params.test_splits:
            split_dataloader = dataloaders[split]
            for i, data in enumerate(split_dataloader):
                model.set_data(data)

                with torch.no_grad():
                    model.forward()
                    output = model.get_output()
                    likelihood.make_likelihood(data, output)

        likelihood.save_likelihood(weight_path.stem)
        model.init_network()      #! NEED?


if __name__ == '__main__':
    set_logger()
    logger.logger.info('\nTest started.\n')

    opt = check_test_options()
    main(opt)

    logger.logger.info('\nTest finished.\n')

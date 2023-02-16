#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import torch
from lib import (
        set_options,
        create_model,
        print_parameter,
        create_dataloader,
        BaseLogger
        )
from lib.component import set_likelihood


logger = BaseLogger.get_logger(__name__)


def main(args):
    model_params = args['model']
    dataloader_params = args['dataloader']
    conf_params = args['conf']
    print_params = args['print']
    print_parameter(print_params)

    test_splits = conf_params.test_splits
    task = conf_params.task
    num_outputs_for_label = conf_params.num_outputs_for_label
    save_datetime_dir = conf_params.save_datetime_dir
    weight_paths = conf_params.weight_paths

    model = create_model(model_params)
    dataloaders = {split: create_dataloader(dataloader_params, split=split) for split in test_splits}
    likelihood = set_likelihood(task, num_outputs_for_label)

    for weight_path in weight_paths:
        logger.info(f"Inference ...")
        model.load_weight(weight_path)
        model.eval()

        for i, split in enumerate(test_splits):
            for j, data in enumerate(dataloaders[split]):
                in_data, _ = model.set_data(data)

                with torch.no_grad():
                    outputs = model(in_data)

                # Make a new likelihood every batch
                df_likelihood = likelihood.make_format(data, outputs)

                if i + j == 0:
                    save_dir = Path(save_datetime_dir, 'likelihoods')
                    save_dir.mkdir(parents=True, exist_ok=True)
                    save_path = Path(save_dir, 'likelihood_' + Path(weight_path).stem + '.csv')
                    df_likelihood.to_csv(save_path, index=False)
                else:
                    df_likelihood.to_csv(save_path, mode='a', index=False, header=False)

        # Reset the current weight by initializing network.
        model.init_network()


if __name__ == '__main__':
    try:
        logger.info('\nTest started.\n')

        args = set_options(phase='test')
        main(args)

    except Exception as e:
        logger.error(e, exc_info=True)

    else:
        logger.info('\nTest finished.\n')

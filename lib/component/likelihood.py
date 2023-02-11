#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import torch
from typing import List, Dict


class Likelihood:
    def __init__(self, task: str, num_outputs_for_label: Dict[str, int]) -> None:
        self.task = task
        self.num_outputs_for_label = num_outputs_for_label
        self.pred_column_list = self._make_pred_columns(self.num_outputs_for_label)
        self.base_column_list = self._set_base_colums(self.task)

    def _set_base_colums(self, task):
        if task == 'classification':
            base_columns = ['uniqID','group', 'imgpath', 'split']
            return base_columns
        elif task == 'regression':
            base_columns = ['uniqID','group', 'imgpath', 'split']
            return base_columns
        elif task == 'deepsurv':
            base_columns = ['uniqID','group', 'imgpath', 'split', 'periods']
            return base_columns
        else:
            raise ValueError(f"Invalid task: {task}.")

    def _make_pred_columns(self, num_outputs_for_label: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Make column names of predictions with label name and its number of classes.

        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label

        Returns:
            Dict[str, List[str]]: label and list of columns of predictions with its class number

        eg.
        {label_A: 2, label_B: 3} -> {label_A: [pred_A_0, pred_A_1], label_B: [pred_B_0, pred_B_1]}
        {label_A: 1, label_B: 1} -> {label_A: [pred_A], label_B: [pred_B]}
        """
        pred_columns = dict()
        for label_name, num_classes in num_outputs_for_label.items():
            _pred_list = []
            if num_classes > 1:
                for ith_class in range(num_classes):
                    _ith_pred = label_name.replace('label', 'pred') + '_' + str(ith_class)
                    _pred_list.append(_ith_pred)
            else:
                _pred = label_name.replace('label', 'pred')
                _pred_list.append(_pred)

            pred_columns[label_name] = _pred_list
        return pred_columns

    def make_likelihood(self, data: Dict, output: Dict[str, torch.Tensor]) -> pd.DataFrame:
            """
            Make DataFrame of likelihood every batch

            Args:
                data (Dict): batch data from dataloader
                output (Dict[str, torch.Tensor]): output of model
            """

            _base = {data[column_name] for column_name in self.base_column_list}
            _df_base = pd.DataFrame(_base)

            if any(data['labels']):
                for label_name, pred in output.items():
                    _df_label = pd.DataFrame({ label_name: [data['labels'][label_name]] })  # Handle one label at a time
                    _pred_columns = self.pred_column_list[label_name]
                    _pred = pred.to('cpu').detach().numpy().copy()
                    _df_pred = pd.DataFrame(_pred, columns=_pred_columns)
                    df_likelihood = pd.concat([_df_base, _df_label, _df_pred], axis=1)
                return df_likelihood
            else:
                for label_name, pred in output.items():
                    _pred_columns = self.pred_column_list[label_name]
                    _pred = pred.to('cpu').detach().numpy().copy()
                    _df_pred = pd.DataFrame(_pred, columns=_pred_columns)
                    df_likelihood = pd.concat([_df_base, _df_pred], axis=1)
                return df_likelihood



def set_likelihood(task: str, num_outputs_for_label: Dict[str, int]) -> Likelihood:
    """
    Set Likelihood

    Args:
        task (str): task
        num_outputs_for_label (Dict[str, int]): number of classes for each label

    Returns:
        Likelihood: Likelihood
    """
    return Likelihood(task, num_outputs_for_label)


#def writeout_likelihood(df_likelihood: pd.DataFrame, save_path: str, times_written: int) -> None:
#        """
#        Write out likelihoood
#
#        Args:
#            save_path (str): sava path of likelihood
#            count (int): number of times written out
#
#        """
#        if times_written > 0:
#            df_likelihood.to_csv(save_path, mode='a', header=False)
#        else:
#            save_dir = Path(save_path).parents[0]
#            save_dir.mkdir(parents=True, exist_ok=True)
#            df_likelihood.to_csv(save_path, index=False)
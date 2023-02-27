#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import torch
from typing import List, Dict


class Likelihood:
    """
    Class for making likelihood.
    """
    def __init__(self, task: str, num_outputs_for_label: Dict[str, int]) -> None:
        """
        Args:
            task (str): task
            num_outputs_for_label (Dict[str, int]): number of classes for each label
        """
        self.task = task
        self.num_outputs_for_label = num_outputs_for_label
        self.base_column_list = self._set_base_columns(self.task)
        self.pred_column_list = self._make_pred_columns(self.task, self.num_outputs_for_label)

    def _set_base_columns(self, task: str) -> List[str]:
        """
        Return base columns.

        Args:
            task (str): task

        Returns:
            List[str]: base columns except columns of label and prediction
        """
        if (task == 'classification') or (task == 'regression'):
            base_columns = ['uniqID', 'group', 'imgpath', 'split']
            return base_columns
        elif task == 'deepsurv':
            base_columns = ['uniqID', 'group', 'imgpath', 'split', 'periods']
            return base_columns
        else:
            raise ValueError(f"Invalid task: {task}.")

    def _make_pred_columns(self, task: str, num_outputs_for_label: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Make column names of predictions with label name and its number of classes.

        Args:
            task (str):  task
            num_outputs_for_label (Dict[str, int]): number of classes for each label

        Returns:
            Dict[str, List[str]]: label and list of columns of predictions with its class number

        eg.
        {label_A: 2, label_B: 2} -> {label_A: [pred_label_A_0, pred_label_A_1], label_B: [pred_label_B_0, pred_label_B_1]}
        {label_A: 1, label_B: 1} -> {label_A: [pred_label_A], label_B: [pred_label_B]}
        """
        pred_columns = dict()
        if task == 'classification':
            for label_name, num_classes in num_outputs_for_label.items():
                pred_columns[label_name] = ['pred_' + label_name + '_' + str(i) for i in range(num_classes)]
            return pred_columns
        elif (task == 'regression') or (task == 'deepsurv'):
            for label_name, num_classes in num_outputs_for_label.items():
                pred_columns[label_name] = ['pred_' + label_name]
            return pred_columns
        else:
            raise ValueError(f"Invalid task: {task}.")

    def make_format(self, data: Dict, output: Dict[str, torch.Tensor]) -> pd.DataFrame:
            """
            Make a new DataFrame of likelihood every batch.

            Args:
                data (Dict): batch data from dataloader
                output (Dict[str, torch.Tensor]): output of model
            """
            _likelihood = {column_name: data[column_name] for column_name in self.base_column_list}
            df_likelihood = pd.DataFrame(_likelihood)

            if any(data['labels']):
                for label_name, pred in output.items():
                    _df_label = pd.DataFrame({label_name: data['labels'][label_name].tolist()})
                    pred = pred.to('cpu').detach().numpy().copy()
                    _df_pred = pd.DataFrame(pred, columns=self.pred_column_list[label_name])
                    df_likelihood = pd.concat([df_likelihood, _df_label, _df_pred], axis=1)
                return df_likelihood
            else:
                for label_name, pred in output.items():
                    pred = pred.to('cpu').detach().numpy().copy()
                    _df_pred = pd.DataFrame(pred, columns=self.pred_column_list[label_name])
                    df_likelihood = pd.concat([df_likelihood, _df_pred], axis=1)
                return df_likelihood


def set_likelihood(task: str, num_outputs_for_label: Dict[str, int]) -> Likelihood:
    """
    Set likelihood.

    Args:
        task (str): task
        num_outputs_for_label (Dict[str, int]): number of classes for each label

    Returns:
            Likelihood: instance of class Likelihood
    """
    return Likelihood(task, num_outputs_for_label)

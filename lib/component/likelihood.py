#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from pathlib import Path
import pandas as pd
import torch
from typing import List, Dict


class Likelihood:
    """
    class for making likeihood
    """
    def __init__(self, task: str, num_outputs_for_label: Dict[str, int]) -> None:
        """
        Args:
            task (str): task
            num_outputs_for_label (Dict[str, int]): number of classes for each label
        """
        self.task = task
        self.num_outputs_for_label = num_outputs_for_label
        self.base_column_list = self._set_base_colums(self.task)
        self.pred_column_list = self._make_pred_columns(self.num_outputs_for_label)

    def _set_base_colums(self, task: str) -> List[str]:
        """_summary_

        Args:
            task (str): task

        Returns:
            List[str]: columns except prediction
        """
        if task == 'classification':
            base_columns = ['uniqID', 'group', 'imgpath', 'split']
            return base_columns
        elif task == 'regression':
            base_columns = ['uniqID', 'group', 'imgpath', 'split']
            return base_columns
        elif task == 'deepsurv':
            base_columns = ['uniqID', 'group', 'imgpath', 'split', 'periods']
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

    def make_form(self, data: Dict, output: Dict[str, torch.Tensor]) -> pd.DataFrame:
            """
            Make DataFrame of likelihood every batch

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

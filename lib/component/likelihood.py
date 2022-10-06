#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from typing import List, Dict, Union
import torch
import numpy


class BaseLikelihood:
    """
    Class for making likelihood
    """
    def __init__(self, class_name_in_raw_label: Dict[str, Dict[str, int]], test_datetime: str) -> None:
        """
        Args:
            class_name_in_raw_label (Dict[str, Dict[str, int]]): class names in each label
            test_datetime (str): date time for test
        """
        self.class_name_in_raw_label = class_name_in_raw_label
        self.test_datetime = test_datetime
        self.df_likelihood = pd.DataFrame()

    def _convert_to_numpy(self, raw_data: torch.Tensor) -> numpy.ndarray:
        """"
        Convert Tensor of output of model to numpy

        Args:
            raw_data (torch.Tensor): output of model

        Returns:
            numpy.ndarray: numpy of output of model to numpy
        """
        converted_data = raw_data.to('cpu').detach().numpy().copy()
        return converted_data

    def _make_pred_names(self, raw_label_name: str) -> List[str]:
        """
        Create column names of predictions with raw label name and class name as suffix.

        Args:
            raw_label_name (str): raw label name

        Returns:
            List[str]: List column names with class name suffix.
        """
        pred_names = []
        class_names = self.class_name_in_raw_label[raw_label_name]
        for class_name in class_names.keys():
            pred_name = 'pred_' + raw_label_name + '_' + str(class_name)
            pred_names.append(pred_name)
        return pred_names

    def make_likehood(
                    self,
                    data: Dict[str, Union[str, int, Dict[str, int], float]],
                    output: Dict[str, torch.Tensor]
                    ) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict[str, Union[str, int, Dict[str, int], float]]): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'ExamID': data['ExamID'],
                            'split': data['split']
                            })

        for internal_label_name, output in output.items():
            # raw_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name]
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # output
            pred_names = self._make_pred_names(raw_label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)

    def save_likelihood(self, save_name: str = None) -> None:
        """
        Save likelihoood.

        Args:
            save_name (str): save name for likelihood
        """
        save_dir = Path('./results/sets', self.test_datetime, 'likelihoods')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'likelihood_' + save_name + '.csv')
        self.df_likelihood.to_csv(save_path, index=False)


class ClsLikelihood(BaseLikelihood):
    """
    Class for likelihood of classification
    """
    def __init__(self, class_name_in_raw_label: Dict[str, Dict[str, int]], test_datetime: str) -> None:
        """
        Args:
            class_name_in_raw_label (Dict[str, Dict[str, int]]): class names in each label
            test_datetime (str): date time for test
        """
        super().__init__(class_name_in_raw_label, test_datetime)


class RegLikelihood(BaseLikelihood):
    """
    Class for likelihood of regression
    """
    def __init__(self, class_name_in_raw_label: Dict[str, Dict[str, int]], test_datetime: str) -> None:
        """
        Args:
            class_name_in_raw_label (Dict[str, Dict[str, int]]): class names in each label
            test_datetime (str): date time for test
        """
        super().__init__(class_name_in_raw_label, test_datetime)

    # Orverwrite
    def _make_pred_names(self, raw_label_name: str) -> List[str]:
        """
        Create column names of predictions with raw label name as suffix.

        Args:
            raw_label_name (str): raw label name

        Returns:
            List[str]: List column names with class name suffix.
        """
        pred_names = []
        pred_names.append('pred_' + raw_label_name)
        return pred_names


class DeepSurvLikelihood(RegLikelihood):
    """
    Class for likelihood of DeepSurv
    """
    def __init__(self, class_name_in_raw_label: Dict[str, Dict[str, int]], test_datetime: str) -> None:
        """
        Args:
            class_name_in_raw_label (Dict[str, Dict[str, int]]): class names in each label
            test_datetime (str): date time for test
        """
        super().__init__(class_name_in_raw_label, test_datetime)

    # Orverwrite
    def make_likehood(
                    self,
                    data: Dict[str, Dict[str, int]],
                    output: Dict[str, torch.Tensor]
                    ) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict[str, Dict[str, int]]): _description_
            output (Dict[str, torch.Tensor]): _description_
        """
        _period_list = self._convert_to_numpy(data['period'])
        _df_new = pd.DataFrame({
                            'Filename': data['Filename'],
                            'Institution': data['Institution'],
                            'ExamID': data['ExamID'],
                            'split': data['split'],
                            'period': [int(period) for period in _period_list]
                            })

        for internal_label_name, output in output.items():
            # raw_label, internal_label
            raw_label_name = internal_label_name.replace('internal_', '')
            _internal_label_list = self._convert_to_numpy(data['internal_labels'][internal_label_name])
            internal_label_list = [int(internal_label) for internal_label in _internal_label_list]
            _df_raw_label = pd.DataFrame({
                                    raw_label_name: data['raw_labels'][raw_label_name],
                                    internal_label_name: internal_label_list
                                    })
            _df_new = pd.concat([_df_new, _df_raw_label], axis=1)

            # output
            pred_names = self._make_pred_names(raw_label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


def set_likelihood(
                task: str,
                class_name_in_raw_label: Dict[str, Dict[str, int]],
                test_datetime: str
                ) -> BaseLikelihood:
    """
    Set likelihood object depending task.

    Args:
        task (str): task
        class_name_in_raw_label (Dict[str, Dict[str, int]]): class names in each label
        test_datetime (str): date time for test

    Returns:
        BaseLikelihood: class of likelihood
    """
    if task == 'classification':
        return ClsLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'regression':
        return RegLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'deepsurv':
        return DeepSurvLikelihood(class_name_in_raw_label, test_datetime)
    else:
        raise ValueError(f"Invalid task: {task}.")

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from typing import List, Dict, Union
import torch
import numpy


class BaseLikelihood:
    """
    Class for making likelihood.
    Substantialy, this is for making likelihood for classification.
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], test_datetime_dirpath: Path) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            test_datetime_dirpath (Path): save path to save likelihood
        """

        self.num_outputs_for_label = num_outputs_for_label
        self.test_datetime_dirpath = test_datetime_dirpath
        self.df_likelihood = pd.DataFrame()

    def _convert_to_numpy(self, output: torch.Tensor) -> numpy.ndarray:
        """"
        Convert Tensor of output of model to numpy

        Args:
            output (torch.Tensor): output of model

        Returns:
            numpy.ndarray: numpy of output of model
        """
        converted_data = output.to('cpu').detach().numpy().copy()
        return converted_data

    def _make_pred_names(self) -> None:
        raise NotImplementedError

    def make_likehood_type(self) -> None:
        raise NotImplementedError

    def save_likelihood(self, test_datetime: Path, save_name: str) -> None:
        """
        Save likelihoood.

        Args:
            test_datetime (Path): directory named datetime to save likelihood
            save_name (str): save name of likelihood
        """
        save_dir = Path(test_datetime, 'likelihoods')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'likelihood_' + save_name).with_suffix('.csv')
        self.df_likelihood.to_csv(save_path, index=False)


class ClsLikelihood(BaseLikelihood):
    """
    Class for likelihood of classification
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], test_datetime_dirpath: Path) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            test_datetime_dirpath (Path): save path to save likelihood
        """
        super().__init__(num_outputs_for_label, test_datetime_dirpath)

    def _make_pred_names(self, label_name: str) -> List[str]:
        """
        Create column names of predictions with label name and class name.

        Args:
            label_name (str): label name

        Returns:
            List[str]: List column names with class name suffix.
        """
        pred_names = []
        _num_outputs = self.num_outputs_for_label[label_name]
        for ith_class in range(_num_outputs):
            pred_name = 'pred_' + label_name + '_' + str(ith_class)
            pred_names.append(pred_name)

        return pred_names

    def make_likehood(self, data: Dict, output: Dict[str, torch.Tensor]) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split']
                            })

        for label_name, output in output.items():
            # label
            _label_list = [int(label) for label in data['labels'][label_name]]
            _df_label = pd.DataFrame({label_name: _label_list})
            # output
            pred_names = self._make_pred_names(label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_label, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


class RegLikelihood(BaseLikelihood):
    """
    Class for likelihood of regression
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], test_datetime_dirpath: Path) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            test_datetime_dirpath (Path): save path to save likelihood
        """
        super().__init__(num_outputs_for_label, test_datetime_dirpath)

    def _make_pred_names(self, label_name: str) -> List[str]:
        """
        Create column names of predictions with label name.

        Args:
            label_name (str): label name

        Returns:
            List[str]: List column names with class name suffix.
        """
        pred_names = []
        pred_names.append('pred_' + label_name)
        return pred_names

    def make_likehood(self, data: Dict, output: Dict[str, torch.Tensor]) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split']
                            })

        for label_name, output in output.items():
            # label
            _label_list = [float(label) for label in data['labels'][label_name]]
            _df_label = pd.DataFrame({label_name: _label_list})
            # output
            pred_names = self._make_pred_names(label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_label, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


class DeepSurvLikelihood(RegLikelihood):
    """
    Class for likelihood of DeepSurv
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], test_datetime_dirpath: Path) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            test_datetime_dirpath (Path): save path to save likelihood
        """
        super().__init__(num_outputs_for_label, test_datetime_dirpath)

    def make_likehood(self, data: Dict, output: Dict[str, torch.Tensor]) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split'],
                            'periods': data['periods']
                            })

        for label_name, output in output.items():
            # label
            _label_list = [float(label) for label in data['labels'][label_name]]
            _df_label = pd.DataFrame({label_name: _label_list})
            # output
            pred_names = self._make_pred_names(label_name)
            _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
            _df_new = pd.concat([_df_new, _df_label, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


def set_likelihood(
                task: str,
                num_outputs_for_label: Dict[str, int],
                test_datetime_dirpath: Path
                ) -> Union[ClsLikelihood, RegLikelihood, DeepSurvLikelihood]:
    """
    Set likelihood object depending task.

    Args:
        task (str): task
        num_outputs_for_label (Dict[str, int]): number of classes for each label
        test_datetime_dirpath (Path): save path to save likelihood

    Returns:
        Union[ClsLikelihood, RegLikelihood, DeepSurvLikelihood]: class of likelihood
    """
    if task == 'classification':
        return ClsLikelihood(num_outputs_for_label, test_datetime_dirpath)
    elif task == 'regression':
        return RegLikelihood(num_outputs_for_label, test_datetime_dirpath)
    elif task == 'deepsurv':
        return DeepSurvLikelihood(num_outputs_for_label, test_datetime_dirpath)
    else:
        raise ValueError(f"Invalid task: {task}.")

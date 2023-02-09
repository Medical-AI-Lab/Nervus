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
    def __init__(self, num_outputs_for_label: Dict[str, int], save_datetime_dir: str) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            save_datetime_dir (str): save path to save likelihood
        """

        self.num_outputs_for_label = num_outputs_for_label
        self.save_datetime_dir = save_datetime_dir

    def init_likelihood(self) -> None:
        """
        Set empty DataFrame to store likelihood.
        """
        self.df_likelihood = pd.DataFrame()



# label_cast_type
#classification, deepsurv: int
# regression: float

"""
_data = {
        'uniqID': uniqID,
        'group': group,
        'imgpath': imgpath,
        'split': split,
        'inputs': inputs_value,
        'image': image,
        'labels': label_dict,
        'periods': periods
        }
"""

# label: int or float
# 



class Likelihood:
    def __init__(self, task: str) -> None:
        self.task = task

        if self.task == 'classification':
            self.label_cast_type = 'int'
            self.base_columns = ['uniqID','group', 'imgpath', 'split']

        elif self.task == 'regression':
            self.label_cast_type = 'float'
            self.base_columns = ['uniqID','group', 'imgpath', 'split']

        elif self.task == 'deepsurv':
            self.label_cast_type = 'int'
            self.base_columns = ['uniqID','group', 'imgpath', 'split', 'periods']

        else:
            raise ValueError(f"Invalid task: {task}.")

    def init_likelihood(self):
        df_likelihood = pd.DataFrame([], columns=self.base_columns)
        return df_likelihood





def writeout_likelihood(df_likelihood: pd.DataFrame, save_path: str, times_written: int) -> None:
        """
        Write out likelihoood

        Args:
            save_path (str): sava path of likelihood
            count (int): number of times written out

        """
        if times_written > 0:
            df_likelihood.to_csv(save_path, mode='a', header=False)
        else:
            save_dir = Path(save_path).parents[0]
            save_dir.mkdir(parents=True, exist_ok=True)
            df_likelihood.to_csv(save_path, index=False)


def _convert_to_numpy(output: torch.Tensor) -> numpy.ndarray:
        """"
        Convert Tensor of output of model to numpy

        Args:
            output (torch.Tensor): output of model

        Returns:
            numpy.ndarray: numpy of output of model
        """
        _converted_data = output.to('cpu').detach().numpy().copy()
        return _converted_data


def _make_pred_columns(label_name: str, num_classes: int) -> List[str]:
    """
    Make column names of predictions with label name and its number of classes.

    Args:
        label_name (str): label name

    Returns:
        List[str]: List column names with class name suffix.
    """
    pred_columns = []
    if num_classes > 1:
        for ith_class in range(num_classes):
            _pred_column = label_name.replace('label', 'pred') + '_' + str(ith_class)
            pred_columns.append(_pred_column)
        return pred_columns
    else:
        _pered_column = label_name.replace('label', 'pred')
        pred_columns.append(_pered_column)
        return pred_columns



def make_likelihood(self, data: Dict, output: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """

        _added = {column_name: data[column_name] for column_name in self.base_columns}
        df_added = pd.DataFrame(_added)

        if any(data['labels']):
            for label_name, output in output.items():
                # Need? label has already cast by CSVParser
                # label cast only when label is passed to loss function, data['labels'] is never cast.
                #_label_list = [int(label) for label in data['labels'][label_name]]    # classification, deepsurv
                #_label_list = [float(label) for label in data['labels'][label_name]]  # regression
                #
                # data['labels']: Dict
                #(Pdb) _data['labels']
                #{'label_dog': 1, 'label_cat': 0}
                # { 'label_dog': [1] }
                _df_label = pd.DataFrame({label_name: [data['labels'][label_name]]})  # Handle one label at a time
                pred_columns = _make_pred_columns(label_name)
                _df_output = pd.DataFrame(_convert_to_numpy(output), columns=pred_columns)
                df_added = pd.concat([df_added, _df_label, _df_output], axis=1)
        else:
            for label_name, output in output.items():
                #  output
                pred_columns = _make_pred_columns(label_name)
                _df_output = pd.DataFrame(_convert_to_numpy(output), columns=pred_columns)
                df_added = pd.concat([df_added, _df_output], axis=1)

        return df_added


"""
df_added = pd.DataFrame({
                    'uniqID': data['uniqID'],
                    'group': data['group'],
                    'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                    'split': data['split']
                    })

# deepsurv
df_added = pd.DataFrame({
                'uniqID': data['uniqID'],
                'group': data['group'],
                'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                'split': data['split'],
                'periods': data['periods']
                })
"""


class ClsLikelihood(BaseLikelihood):
    """
    Class for likelihood of classification
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], save_datetime_dir: str) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            save_datetime_dir (str): directory for saving likelihood
        """
        super().__init__(num_outputs_for_label, save_datetime_dir)

    def make_likelihood(self, data: Dict, output: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        df_added = pd.DataFrame({
                            'uniqID': data['uniqID'],
                            'group': data['group'],
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split']
                            })

        if any(data['labels']):
            for label_name, output in output.items():
                # label
                _label_list = [int(label) for label in data['labels'][label_name]]
                _df_label = pd.DataFrame({label_name: _label_list})
                # output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                df_added = pd.concat([df_added, _df_label, _df_output], axis=1)
        else:
            for label_name, output in output.items():
                #  output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                df_added = pd.concat([df_added, _df_output], axis=1)

        #self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)
        return df_added


class RegLikelihood(BaseLikelihood):
    """
    Class for likelihood of regression
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], save_datetime_dir: str) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            save_datetime_dir (str): directory for saving likelihood
        """
        super().__init__(num_outputs_for_label, save_datetime_dir)


    def make_likelihood(self, data: Dict, output: Dict[str, torch.Tensor]) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'uniqID': data['uniqID'],
                            'group': data['group'],
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split']
                            })

        if any(data['labels']):
            for label_name, output in output.items():
                # label
                _label_list = [float(label) for label in data['labels'][label_name]]
                _df_label = pd.DataFrame({label_name: _label_list})
                # output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                _df_new = pd.concat([_df_new, _df_label, _df_output], axis=1)
        else:
            for label_name, output in output.items():
                #  output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                _df_new = pd.concat([_df_new, _df_output], axis=1)

        #self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)


        """ classification
        if any(data['labels']):
            for label_name, output in output.items():
                # label
                _label_list = [int(label) for label in data['labels'][label_name]]
                _df_label = pd.DataFrame({label_name: _label_list})
                # output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                df_added = pd.concat([df_added, _df_label, _df_output], axis=1)
        else:
            for label_name, output in output.items():
                #  output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                df_added = pd.concat([df_added, _df_output], axis=1)
        """

        return df_added

class DeepSurvLikelihood(RegLikelihood):
    """
    Class for likelihood of DeepSurv
    """
    def __init__(self, num_outputs_for_label: Dict[str, int], save_datetime_dir: str) -> None:
        """
        Args:
            num_outputs_for_label (Dict[str, int]): number of classes for each label
            save_datetime_dir (str): directory for saving likelihood
        """
        super().__init__(num_outputs_for_label, save_datetime_dir)

    def make_likelihood(self, data: Dict, output: Dict[str, torch.Tensor]) -> None:
        """
        Make DataFrame of likelihood every batch

        Args:
            data (Dict): batch data from dataloader
            output (Dict[str, torch.Tensor]): output of model
        """
        _df_new = pd.DataFrame({
                            'uniqID': data['uniqID'],
                            'group': data['group'],
                            'imgpath': [str(imgpath) for imgpath in data['imgpath']],
                            'split': data['split'],
                            'periods': data['periods']
                            })

        if any(data['labels']):
            for label_name, output in output.items():
                # label
                _label_list = [int(label) for label in data['labels'][label_name]]
                _df_label = pd.DataFrame({label_name: _label_list})
                # output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                _df_new = pd.concat([_df_new, _df_label, _df_output], axis=1)
        else:
            for label_name, output in output.items():
                #  output
                pred_names = self._make_pred_names(label_name)
                _df_output = pd.DataFrame(self._convert_to_numpy(output), columns=pred_names)
                _df_new = pd.concat([_df_new, _df_output], axis=1)

        self.df_likelihood = pd.concat([self.df_likelihood, _df_new], ignore_index=True)



def set_likelihood(
                task: str,
                num_outputs_for_label: Dict[str, int],
                save_datetime_dir: str
                ) -> Union[ClsLikelihood, RegLikelihood, DeepSurvLikelihood]:
    """
    Set likelihood object depending task.

    Args:
        task (str): task
        num_outputs_for_label (Dict[str, int]): number of classes for each label
        save_datetime_dir (str): directory for saving likelihood

    Returns:
        Union[ClsLikelihood, RegLikelihood, DeepSurvLikelihood]: class of likelihood
    """
    if task == 'classification':
        return ClsLikelihood(num_outputs_for_label, save_datetime_dir)
    elif task == 'regression':
        return RegLikelihood(num_outputs_for_label, save_datetime_dir)
    elif task == 'deepsurv':
        return DeepSurvLikelihood(num_outputs_for_label, save_datetime_dir)
    else:
        raise ValueError(f"Invalid task: {task}.")

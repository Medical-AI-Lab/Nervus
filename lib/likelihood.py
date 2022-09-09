#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
from .logger import get_logger


log = get_logger('models.likelihood')


class BaseLikelihood:
    """
    Class for making likelihood
    """
    def __init__(self, class_name_in_raw_label, test_datetime):
        self.class_name_in_raw_label = class_name_in_raw_label
        self.test_datetime = test_datetime
        self.df_likelihood = pd.DataFrame()

    def _convert_to_numpy(self, raw_data):
        converted_data = raw_data.to('cpu').detach().numpy().copy()
        return converted_data

    def _make_pred_names(self, raw_label_name):
        pred_names = []
        class_names = self.class_name_in_raw_label[raw_label_name]
        for class_name in class_names.keys():
            pred_name = 'pred_' + raw_label_name + '_' + class_name
            pred_names.append(pred_name)
        return pred_names

    def make_likehood(self, data, output):
        """
        Make DataFrame of likelihood every batch

        Args:
            data (dict): batch data from dataloader
            output (dict): output of model
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

    def save_likelihood(self, save_name=None):
        save_dir = Path('./results/sets', self.test_datetime, 'likelihoods')
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir, 'likelihood_' + save_name + '.csv')
        self.df_likelihood.to_csv(save_path, index=False)


class ClsLikelihood(BaseLikelihood):
    """
    Class for likelihood of classification
    This class is exactly the same as BaseLikelihood

    Args:
        BaseLikelihood: Base class for likelihood
    """
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)


class RegLikelihood(BaseLikelihood):
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)

    # Orverwrite
    def _make_pred_names(self, raw_label_name):
        pred_names = []
        pred_names.append('pred_' + raw_label_name)
        return pred_names


class DeepSurvLikelihood(RegLikelihood):
    def __init__(self, class_name_in_raw_label, test_datetime):
        super().__init__(class_name_in_raw_label, test_datetime)

    # Orverwrite
    def make_likehood(self, data, output):
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


def set_likelihood(task, class_name_in_raw_label, test_datetime):
    if task == 'classification':
        return ClsLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'regression':
        return RegLikelihood(class_name_in_raw_label, test_datetime)
    elif task == 'deepsurv':
        return DeepSurvLikelihood(class_name_in_raw_label, test_datetime)
    else:
        log.error(f"Invalid task:{task}.")

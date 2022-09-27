#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple, Union
import argparse


class SplitProvider:
    """
    Class to make label for each class and cast tabular data
    """
    def __init__(self, split_path: str, task: str) -> None:
        """
        Args:
            split_path (Path): path to csv
            task (str): task
        """
        self.split_path = Path(split_path)
        self.task = task

        _df_source = pd.read_csv(self.split_path)
        _df_source_excluded = _df_source[_df_source['split'] != 'exclude'].copy()
        _df_source_labeled, _class_name_in_raw_label = self._make_labelling(_df_source_excluded, self.task)
        self.df_source = self._cast_csv(_df_source_labeled, self.task)

        self.raw_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith('label')])
        self.internal_label_list = list(self.df_source.columns[self.df_source.columns.str.startswith('internal_label')])
        self.class_name_in_raw_label = _class_name_in_raw_label
        self.num_classes_in_internal_label = self._define_num_classes_in_internal_label(self.df_source, self.task)
        self.input_list = list(self.df_source.columns[self.df_source.columns.str.startswith('input')])

        if self.task == 'deepsurv':
            self.period_column = list(self.df_source.columns[self.df_source.columns.str.startswith('period')])[0]
        else:
            self.period_column = None

    # Labeling
    def _make_labelling(self, df_source_excluded: pd.DataFrame, task: str) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        Assign a number to the class name within each label

        Args:
            df_source_excluded (DataFrame): DataFrame of csv without 'exlucde'
            task (str): task

        Returns:
            pd.DataFrame: DataFrame with columns assigned a number to the class name within each label,
            Dict[str, int]: Dictionary with numbers assigned to class names within each label
        """
        # Make dict for labeling
        # _class_name_in_raw_label =
        # {'label_1':{'A':0, 'B':1}, 'label_2':{'C':0, 'D':1, 'E':2}, ...}   classification
        # {'label_1':{}, 'label_2':{}, ...}                                  regression
        # {'label_1':{'A':0, 'B':1}}                                         deepsurv, should be 2 classes only
        _df_tmp = df_source_excluded.copy()
        _raw_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('label')])
        _class_name_in_raw_label = {}
        for raw_label_name in _raw_label_list:
            class_list = _df_tmp[raw_label_name].value_counts().index.tolist()  # {'A', 'B', ... } DECENDING ORDER
            _class_name_in_raw_label[raw_label_name] = {}
            if (task == 'classification') or (task == 'deepsurv'):
                for i, ith_class in enumerate(class_list):
                    _class_name_in_raw_label[raw_label_name][ith_class] = i
            else:
                _class_name_in_raw_label[raw_label_name] = {}  # No need of labeling

        # Labeling
        for raw_label_name, class_name_in_raw_label in _class_name_in_raw_label.items():
            _internal_label = raw_label_name.replace('label', 'internal_label')  # label_XXX -> internal_label_XXX
            if (task == 'classification') or (task == 'deepsurv'):
                for class_name, ground_truth in class_name_in_raw_label.items():
                    _df_tmp.loc[_df_tmp[raw_label_name] == class_name, _internal_label] = ground_truth
            else:
                # When regression
                _df_tmp[_internal_label] = _df_tmp[raw_label_name].copy()    # Just copy. Needed because internal_label_XXX will be cast later.

        _df_source_labeled = _df_tmp.copy()
        return _df_source_labeled, _class_name_in_raw_label

    def _define_num_classes_in_internal_label(self, df_source: pd.DataFrame, task: str) -> Dict[str, int]:
        """
        Find the number of classes for each internal label

        Args:
            df_source (pd.DataFrame): DataFrame of csv
            task (str): task

        Returns:
            Dict[str, int]: Number of classes for each internal label
        """
        # _num_classes_in_internal_label =
        # {internal_label_output_1: 2, internal_label_output_2: 3, ...}   classification
        # {internal_label_output_1: 1, internal_label_output_2: 1, ...}   regression,  should be 1
        # {internal_label_output_1: 1}                                    deepsurv,    should be 1
        _num_classes_in_internal_label = {}
        _internal_label_list = list(df_source.columns[df_source.columns.str.startswith('internal_label')])
        for internal_label_name in _internal_label_list:
            if task == 'classification':
                # Actually _num_classes_in_internal_label can be made from self.class_name_in_raw_label, however
                # it might be natural to count the number of classes in each internal label.
                _num_classes_in_internal_label[internal_label_name] = df_source[internal_label_name].nunique()
            else:
                # When regression or deepsurv
                _num_classes_in_internal_label[internal_label_name] = 1

        return _num_classes_in_internal_label

    # Cast
    def _cast_csv(self, df_source_labeled: pd.DataFrame, task: str) -> pd.DataFrame:
        """
        Cast columns as required by the task.

        Args:
            df_source_labeled (pd.DataFrame): DataFrame of labeled csv
            task (str): task

        Returns:
            pd.DataFrame: cast DataFrame of cvs with labeling
        """
        # label_* : int
        # input_* : float
        _df_tmp = df_source_labeled.copy()
        _input_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('input')])
        _internal_label_list = list(_df_tmp.columns[_df_tmp.columns.str.startswith('internal_label')])

        _cast_input_dict = {input: float for input in _input_list}

        if task == 'classification':
            _cast_internal_label_dict = {internal_label: int for internal_label in _internal_label_list}
        else:
            # When regression or deepsurv
            _cast_internal_label_dict = {internal_label: float for internal_label in _internal_label_list}

        _df_tmp = _df_tmp.astype(_cast_input_dict)
        _df_tmp = _df_tmp.astype(_cast_internal_label_dict)
        _df_casted = _df_tmp.copy()
        return _df_casted


def make_split_provider(split_path: Path, task: str) -> SplitProvider:
    """
    Format csv by making label dependinf on task.

    Args:
        split_path (Path): path to csv
        task (str): task

    Returns:
        SplitProvider: Object to DataFrame of labeled csv
    """
    sp = SplitProvider(split_path, task)
    return sp


#
# The below is for dataloader.
#
class PrivateAugment(torch.nn.Module):
    """
    Augmentation defined privately.
    Variety of augmentation can be written in this class if necessary.
    """
    # For X-ray photo.
    xray_augs_list = [
                    transforms.RandomAffine(degrees=(-3, 3), translate=(0.02, 0.02)),
                    transforms.RandomAdjustSharpness(sharpness_factor=2),
                    transforms.RandomAutocontrast()
                    ]


class InputDataMixin:
    """
    Class to normalizes input data.
    """
    def _make_scaler(self) -> MinMaxScaler:
        """
        Normalizes inputa data by min-max normalization with train data.

        Returns:
            MinMaxScaler: scaler
        """
        _scaler = MinMaxScaler()
        _df_train = self.df_source[self.df_source['split'] == 'train']  # should be normalized with min and max of training data
        _ = _scaler.fit(_df_train[self.input_list])                     # fit only
        return _scaler

    def _load_input_value_if_mlp(self, idx: int) -> Union[torch.Tensor, str]:
        """
        Load input values after converting them into tensor if MLP is used.

        Args:
            idx (int): index

        Returns:
            Union[torch.Tensor[float], str]: tensor of input values, or empty string
        """
        inputs_value = ''

        if self.args.mlp is None:
            return inputs_value

        index_input_list = [self.col_index_dict[input] for input in self.input_list]

        # When specifying iloc[[idx], index_input_list], pd.DataFrame is obtained,
        # it fits the input type of self.scaler.transform.
        # However, after normalizing, the shape of inputs_value is (1, N), where N is the number of input value.
        # so, convert (1, N) -> (N,) by squeeze() so that calculating loss would work.
        _df_inputs_value = self.df_split.iloc[[idx], index_input_list]
        inputs_value = self.scaler.transform(_df_inputs_value).squeeze()  # normalize and squeeze() cenverts (1, 46) -> (46,)
        inputs_value = np.array(inputs_value, dtype=np.float64)
        inputs_value = torch.from_numpy(inputs_value.astype(np.float32)).clone()  # numpy -> Tensor
        return inputs_value


class ImageMixin:
    """
    Class to normalizes and transforms image
    """
    def _make_augmentations(self) -> List:
        """
        Define which augmentation is applied.

        When traning, augmentation is needed for train data only.
        When test, no need of augmentation.
        """
        assert (self.args.augmentation is not None), 'Specify augmentation.'

        _augmentation = []
        if (self.args.isTrain) and (self.split == 'train'):
            if self.args.augmentation == 'xrayaug':
                _augmentation = PrivateAugment.xray_augs_list
            elif self.args.augmentation == 'trivialaugwide':
                _augmentation.append(transforms.TrivialAugmentWide())
            elif self.args.augmentation == 'randaug':
                _augmentation.append(transforms.RandAugment())
            else:
                # ie. self.args.augmentation == 'no':
                pass

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation

    def _make_transforms(self) -> List:
        """
        Make list of transforms.

        Returns:
            list of transforms: image normalization
        """
        _transforms = []
        _transforms.append(transforms.ToTensor())

        assert (self.args.normalize_image is not None), 'Specify normalize_image by yes or no.'
        assert (self.args.in_channel is not None), 'Speficy in_channel by 1 or 3.'
        if self.args.normalize_image == 'yes':
            # transforms.Normalize accepts only Tensor.
            if self.args.in_channel == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))
            else:
                # ie. self.args.in_channel == 3
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        else:
            # ie. self.args.normalize_image == 'no'
            pass

        _transforms = transforms.Compose(_transforms)
        return _transforms

    def _load_image_if_cnn(self, idx: int) -> Union[torch.Tensor, str]:
        """
        Load image and convert it to tensor if any of CNN or ViT is used.

        Args:
            idx (int): index

        Returns:
            Union[torch.Tensor[float], str]: tensor converted from image, or empty string
        """
        image = ''

        if self.args.net is None:
            return image

        assert (self.args.image_dir is not None), 'Specify image_dir.'
        filepath = self.df_split.iat[idx, self.col_index_dict['filepath']]
        image_path = Path(self.args.image_dir, filepath)

        assert (self.args.in_channel is not None), 'Speficy in_channel by 1 or 3.'
        if self.args.in_channel == 1:
            image = Image.open(image_path).convert('L')
        else:
            # ie. self.args.in_channel == 3
            image = Image.open(image_path).convert('RGB')

        image = self.augmentation(image)
        image = self.transform(image)
        return image


class DeepSurvMixin:
    """
    Class to handle required data for deepsurv
    """
    def _load_periods_if_deepsurv(self, idx: int) -> Union[int, str]:
        """
        Return period if deepsurv.

        Args:
            idx (int): index

        Returns:
            Union[int, str]: period, or empty string
        """
        period = ''

        if self.args.task != 'deepsurv':
            return period

        assert (self.args.task == 'deepsurv') and (len(self.internal_label_list)==1), 'Deepsurv cannot work in multi-label.'
        period_key = [key for key in self.col_index_dict if key.startswith('period')][0]
        period = self.df_split.iat[idx, self.col_index_dict[period_key]]
        period = np.array(period, dtype=np.float64)
        period = torch.from_numpy(period.astype(np.float32)).clone()
        return period


class DataSetWidget(InputDataMixin, ImageMixin, DeepSurvMixin):
    """
    Class for a widget to inherit multiple classes simultaneously.
    """
    pass


class LoadDataSet(Dataset, DataSetWidget):
    """
    Dataset for split.
    """
    def __init__(self, args: argparse.Namespace, sp: SplitProvider, split: str) -> None:
        """
        Args:
            args (argparse.Namespace): options
            sp (SplitProvider): Object of Splitprovider
            split (str): split
        """
        super().__init__()

        self.args = args
        self.sp = sp
        self.split = split

        self.df_source = self.sp.df_source
        self.df_split = self.df_source[self.df_source['split'] == self.split]

        self.raw_label_list = self.sp.raw_label_list
        self.internal_label_list = self.sp.internal_label_list
        self.input_list = self.sp.input_list

        self.col_index_dict = {col_name: self.df_split.columns.get_loc(col_name) for col_name in self.df_split.columns}

        if (self.args.mlp is not None):
            self.scaler = self._make_scaler()

        if (self.args.net is not None):
            self.augmentation = self._make_augmentations()
            self.transform = self._make_transforms()

    def __len__(self) -> int:
        """
        Return length of DataFrame.

        Returns:
            int: length of DataFrame
        """
        return len(self.df_split)

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, Dict[str, int], float]]:
        """
        Return data row specified by index.

        Args:
            idx (int): index

        Returns:
            Dict[str, Union[str, int, Dict[str, int], float]]: dictionary of data
        """
        filename = Path(self.df_split.iat[idx, self.col_index_dict['filepath']]).name
        examid = self.df_split.iat[idx, self.col_index_dict['ExamID']]
        institution = self.df_split.iat[idx, self.col_index_dict['Institution']]
        raw_label_dict = {raw_label_name: self.df_split.iat[idx, self.col_index_dict[raw_label_name]] for raw_label_name in self.raw_label_list}
        internal_label_dict = {internal_label_name: self.df_split.iat[idx, self.col_index_dict[internal_label_name]] for internal_label_name in self.internal_label_list}
        inputs_value = self._load_input_value_if_mlp(idx)
        image = self._load_image_if_cnn(idx)
        period = self._load_periods_if_deepsurv(idx)
        split = self.df_split.iat[idx, self.col_index_dict['split']]
        return {
                'Filename': filename,
                'ExamID': examid,
                'Institution': institution,
                'raw_labels': raw_label_dict,
                'internal_labels': internal_label_dict,
                'inputs': inputs_value,
                'image': image,
                'period': period,
                'split': split
                }


def _make_sampler(split_data: LoadDataSet) -> WeightedRandomSampler:
    """
    Make sampler.

    Args:
        split_data (LoadDataSet): dataset for anyt of train

    Returns:
        WeightedRandomSampler: sampler
    """
    _target = []
    for _, data in enumerate(split_data):
        _target.append(list(data['internal_labels'].values())[0])

    class_sample_count = np.array([len(np.where(_target == t)[0]) for t in np.unique(_target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in _target])
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler


def create_dataloader(args: argparse.Namespace, sp: SplitProvider, split: str = None) -> DataLoader:
    """
    Creeate data loader ofr split.

    Args:
        args (argparse.Namespace): options
        sp (SplitProvider): Object of SplitProvider
        split (str, optional): split. Defaults to None.

    Returns:
        DataLoader: data loader
    """
    split_data = LoadDataSet(args, sp, split)

    # args never has both 'batch_size' and 'test_batch_size'.
    if args.isTrain:
        batch_size = args.batch_size
    else:
        batch_size = args.test_batch_size

    assert (args.sampler is not None), 'Specify sampler by yes or no.'
    if args.sampler == 'yes':
        assert ((args.task == 'classification') or (args.task == 'deepsurv')), 'Cannot make sampler in regression.'
        assert (len(sp.raw_label_list) == 1), 'Cannot make sampler for multi-label.'
        shuffle = False
        sampler = _make_sampler(split_data)
    else:
        # ie. args.sampler == 'no'
        shuffle = True
        sampler = None

    split_loader = DataLoader(
                            dataset=split_data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            sampler=sampler
                            )
    return split_loader

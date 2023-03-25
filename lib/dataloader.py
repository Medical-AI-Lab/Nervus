#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import pickle
from .logger import BaseLogger
from typing import List, Dict, Union
import pandas as pd


logger = BaseLogger.get_logger(__name__)


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
        Make scaler to normalize input data by min-max normalization with train data.

        Returns:
            MinMaxScaler: scaler
        """
        scaler = MinMaxScaler()
        _df_train = self.df_source[self.df_source['split'] == 'train']  # should be normalized with min and max of training data
        _ = scaler.fit(_df_train[self.input_list])                      # fit only
        return scaler

    def save_scaler(self, save_path :str) -> None:
        """
        Save scaler

        Args:
            save_path (str): path for saving scaler.
        """
        #save_scaler_path = Path(save_datetime_dir, 'scaler.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_scaler(self, scaler_path :str) -> None:
        """
        Load scaler.

        Args:
            scaler_path (str): path to scaler
        """
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler

    def _normalize_inputs(self, df_inputs: pd.DataFrame) -> torch.FloatTensor:
        """
        Normalize inputs.

        Args:
            df_inputs (pd.DataFrame): DataFrame of inputs

        Returns:
            torch.FloatTensor: normalized inputs

        Note:
        After iloc[[idx], index_input_list], pd.DataFrame is obtained.
        DataFrame fits the input type of self.scaler.transform.
        However, after normalizing, the shape of inputs_value is (1, N), where N is the number of input values.
        Since the shape (1, N) is not acceptable when forwarding, convert (1, N) -> (N,) is needed.
        """
        inputs_value = self.scaler.transform(df_inputs).reshape(-1)  #    np.float64
        inputs_value = np.array(inputs_value, dtype=np.float32)      # -> np.float32
        inputs_value = torch.from_numpy(inputs_value).clone()        # -> torch.float32
        return inputs_value

    def _load_input_value_if_mlp(self, idx: int) -> Union[torch.FloatTensor, str]:
        """
        Load input values after converting them into tensor if MLP is used.

        Args:
            idx (int): index

        Returns:
            Union[torch.Tensor[float], str]: tensor of input values, or empty string
        """
        inputs_value = ''

        if self.params.mlp is None:
            return inputs_value

        index_input_list = [self.col_index_dict[input] for input in self.input_list]
        _df_inputs = self.df_split.iloc[[idx], index_input_list]
        inputs_value = self._normalize_inputs( _df_inputs)
        return inputs_value


class ImageMixin:
    """
    Class to normalize and transform image.
    """
    def _make_augmentations(self) -> List:
        """
        Define which augmentation is applied.

        When training, augmentation is needed for train data only.
        When test, no need of augmentation.
        """
        _augmentation = []
        if (self.params.isTrain) and (self.split == 'train'):
            if self.params.augmentation == 'xrayaug':
                _augmentation = PrivateAugment.xray_augs_list
            elif self.params.augmentation == 'trivialaugwide':
                _augmentation.append(transforms.TrivialAugmentWide())
            elif self.params.augmentation == 'randaug':
                _augmentation.append(transforms.RandAugment())
            else:
                # ie. self.params.augmentation == 'no':
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

        if self.params.normalize_image == 'yes':
            # transforms.Normalize accepts only Tensor.
            if self.params.in_channel == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))
            else:
                # ie. self.params.in_channel == 3
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        _transforms = transforms.Compose(_transforms)
        return _transforms

    def _open_image_in_channel(self, imgpath: str, in_channel: int) -> Image:
        """
        Open image in channel.

        Args:
            imgpath (str): path to image
            in_channel (int): channel, or 1 or 3

        Returns:
            Image: PIL image
        """
        if in_channel == 1:
            image = Image.open(imgpath).convert('L')    # eg. np.array(image).shape = (64, 64)
            return image
        else:
            # ie. self.params.in_channel == 3
            image = Image.open(imgpath).convert('RGB')  # eg. np.array(image).shape = (64, 64, 3)
            return image

    def _load_image_if_cnn(self, idx: int) -> Union[torch.Tensor, str]:
        """
        Load image and convert it to tensor if any of CNN or ViT is used.

        Args:
            idx (int): index

        Returns:
            Union[torch.Tensor[float], str]: tensor converted from image, or empty string
        """
        image = ''

        if self.params.net is None:
            return image

        imgpath = self.df_split.iat[idx, self.col_index_dict['imgpath']]
        image = self._open_image_in_channel(imgpath, self.params.in_channel)
        image = self.augmentation(image)
        image = self.transform(image)
        return image


class DeepSurvMixin:
    """
    Class to handle required data for deepsurv.
    """
    def _load_periods_if_deepsurv(self, idx: int) -> Union[torch.FloatTensor, str]:
        """
        Return period if deepsurv.

        Args:
            idx (int): index

        Returns:
            Union[torch.FloatTensor, str]: period, or empty string
        """
        periods = ''

        if self.params.task != 'deepsurv':
            return periods

        assert (self.params.task == 'deepsurv') and (len(self.label_list) == 1), 'Deepsurv cannot work in multi-label.'
        periods = self.df_split.iat[idx, self.col_index_dict[self.period_name]]  #    int64
        periods = np.array(periods, dtype=np.float32)                            # -> np.float32
        periods = torch.from_numpy(periods).clone()                              # -> torch.float32
        return periods


class DataSetWidget(InputDataMixin, ImageMixin, DeepSurvMixin):
    """
    Class for a widget to inherit multiple classes simultaneously.
    """
    pass


class LoadDataSet(Dataset, DataSetWidget):
    """
    Dataset for split.
    """
    def __init__(
                self,
                params,
                split: str
                ) -> None:
        """
        Args:
            params (ParamSet): parameter for model
            split (str): split
        """
        self.params = params
        self.df_source = self.params.df_source
        self.split = split

        self.input_list = self.params.input_list
        self.label_list = self.params.label_list

        if self.params.task == 'deepsurv':
            self.period_name = self.params.period_name

        self.df_split = self.df_source[self.df_source['split'] == self.split]
        self.col_index_dict = {col_name: self.df_split.columns.get_loc(col_name) for col_name in self.df_split.columns}

        # For input data
        if self.params.mlp is not None:
            assert (self.input_list != []), f"input list is empty."
            if params.isTrain:
                self.scaler = self._make_scaler()
            else:
                # load scaler used at training.
                self.scaler = self.load_scaler(self.params.scaler_path)

        # For image
        if self.params.net is not None:
            self.augmentation = self._make_augmentations()
            self.transform = self._make_transforms()

    def __len__(self) -> int:
        """
        Return length of DataFrame.

        Returns:
            int: length of DataFrame
        """
        return len(self.df_split)

    def _load_label(self, idx: int) -> Dict[str, Union[int, float]]:
        """
        Return labels.
        If no column of label when csv of external dataset is used,
        empty dictionary is returned.

        Args:
            idx (int): index

        Returns:
            Dict[str, Union[int, float]]: dictionary of label name and its value
        """
        # For checking if columns of labels exist when used csv for external dataset.
        label_list_in_split = list(self.df_split.columns[self.df_split.columns.str.startswith('label')])
        label_dict = dict()
        if label_list_in_split != []:
            for label_name in self.label_list:
                label_dict[label_name] = self.df_split.iat[idx, self.col_index_dict[label_name]]
        else:
            # no label
            pass
        return label_dict

    def __getitem__(self, idx: int) -> Dict:
        """
        Return data row specified by index.

        Args:
            idx (int): index

        Returns:
            Dict: dictionary of data to be passed model
        """
        uniqID = self.df_split.iat[idx, self.col_index_dict['uniqID']]
        group = self.df_split.iat[idx, self.col_index_dict['group']]
        imgpath = self.df_split.iat[idx, self.col_index_dict['imgpath']]
        split = self.df_split.iat[idx, self.col_index_dict['split']]
        inputs_value = self._load_input_value_if_mlp(idx)
        image = self._load_image_if_cnn(idx)
        label_dict = self._load_label(idx)
        periods = self._load_periods_if_deepsurv(idx)

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
        return _data


class DistributedWeightedSampler:
    def __init__(self, split_data, shuffle):
        raise NotImplementedError("DistributedWeightedSampler")


class SamplerMaker:
    """
    Class to define sampler.
    """
    @classmethod
    def set_distributed_sampler(
                                cls,
                                split_data: LoadDataSet = None,
                                shuffle: bool = None
                                ) -> DistributedSampler:
        """
        Set DistributedSampler.

        Args:
            split_data (LoadDataSet): dataset
            shuffle (bool): whether shuffle pr not

        Returns:
            DistributedSampler: sampler
        """
        _sampler = DistributedSampler(
                                    split_data,
                                    shuffle=shuffle
                                    )
        return _sampler

    @classmethod
    def set_weightedrandom_sampler(
                                cls,
                                split_data: LoadDataSet = None
                                ) -> WeightedRandomSampler:
        """
        Set WeightedRandomSampler.

        Args:
            split_data (LoadDataSet): dataset

        Returns:
            WeightedRandomSampler: sampler

        Note:
            WeightedRandomSampler does shuffle automatically.
        """
        _target = []
        for _, data in enumerate(split_data):
            _target.append(list(data['labels'].values())[0])

        class_sample_count = np.array([len(np.where(_target == t)[0]) for t in np.unique(_target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in _target])

        _sampler = WeightedRandomSampler(
                                        samples_weight,
                                        len(samples_weight)
                                        )
        return _sampler

    @classmethod
    def set_weighted_random_distributed_sampler(
                                                cls,
                                                split_data: LoadDataSet = None,
                                                shuffle=None
                                                ) -> DistributedWeightedSampler:
        _sampler = DistributedWeightedSampler(
                                            split_data,
                                            shuffle
                                            )
        return _sampler


def set_sampler(
                task: str = None,
                label_list: List[str] = None,
                sampler: str = None,
                split_data: LoadDataSet = None
                ) -> Union[DistributedSampler, WeightedRandomSampler, DistributedWeightedSampler]:
    """
    Set sampler.

    Args:
        task (str): task
        label_list (List[str]): label list
        sampler (str): sampler
        split_data (LoadDataSet): dataset

    Returns:
        Union[DistributedSampler, WeightedRandomSampler, DistributedWeightedSampler]: sampler
    """
    _sampler = None

    if sampler == 'no':
        return _sampler

    if sampler == 'distributed':
        _sampler = SamplerMaker.set_distributed_sampler(split_data=split_data, shuffle=True)
        return _sampler

    if 'weight' in sampler:
        # WeightedRandomSampler does shuffle automatically.
        assert (task == 'classification') or (task == 'deepsurv'), 'Cannot make sampler in regression.'
        assert (len(label_list) == 1), 'Cannot make weighted sampler for multi-label.'

        if sampler == 'weighted':
            _sampler = SamplerMaker.set_weightedrandom_sampler(split_data=split_data)
            return _sampler

        if sampler == 'distweight':
            _sampler = SamplerMaker.set_weighted_random_distributed_sampler(split_data=split_data, shuffle=True)
            return _sampler

    raise ValueError(f"Invalid sampler: {sampler}.")


def create_dataloader(
                    params,
                    split: str = None
                    ) -> DataLoader:
    """
    Create data loader for split.

    Args:
        params (ParamSet): parameter for dataloader
        split (str): split.

    Returns:
        DataLoader: data loader
    """
    split_data = LoadDataSet(params, split)

    # sampler
    if params.isTrain:
        _sampler = set_sampler(
                            task=params.task,
                            label_list=params.label_list,
                            sampler=params.sampler,
                            split_data=split_data
                            )
    else:
        # No sampler at test.
        _sampler = None

    # shuffle
    if params.isTrain:
        if _sampler is None:
            shuffle = True
        else:
            # When using sampler at training, whether shuffle or not is defined in sampler.
            shuffle = False
    else:
        # No shuffle at test.
        shuffle = False

    # batch size
    if params.isTrain:
        batch_size = params.batch_size
    else:
        batch_size = params.test_batch_size

    # ---------------------
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # ---------------------
    split_loader = DataLoader(
                            dataset=split_data,
                            batch_size=batch_size,
                            num_workers=os.cpu_count(),
                            sampler=_sampler,
                            shuffle=shuffle,
                            pin_memory=True
                            )
    return split_loader

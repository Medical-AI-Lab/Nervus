#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
from .logger import Logger as logger
from typing import List, Dict, Union
import argparse
from .env import SplitProvider


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
        _augmentation = []
        if (self.args.isTrain) and (self.split == 'train'):
            if self.args.augmentation == 'xrayaug':
                _augmentation = PrivateAugment.xray_augs_list
            elif self.args.augmentation == 'trivialaugwide':
                _augmentation.append(transforms.TrivialAugmentWide())
            elif self.args.augmentation == 'randaug':
                _augmentation.append(transforms.RandAugment())
            elif self.args.augmentation == 'no':
                pass
            else:
                logger.logger.error(f"Invalid augmentation for {self.split}: {self.args.augmentation}.")
                exit()

        _augmentation = transforms.Compose(_augmentation)
        return _augmentation

    def _make_transforms(self) -> List:
        """
        Make list of transforms.

        Returns:
            list of transforms: image normalization
        """
        assert ((self.args.in_channel == 1) or (self.args.in_channel == 3)), f"Invalid input channel: {self.args.in_channel}."

        _transforms = []
        _transforms.append(transforms.ToTensor())

        if self.args.normalize_image == 'yes':
            # transforms.Normalize accepts only Tensor.
            if self.args.in_channel == 1:
                _transforms.append(transforms.Normalize(mean=(0.5, ), std=(0.5, )))
            else:
                _transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        elif self.args.normalize_image == 'no':
            pass
        else:
            logger.logger.error(f"Invalid normalize_image: {self.args.augmentation}.")
            exit()

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

        if self.args.in_channel == 1:
            image = Image.open(image_path).convert('L')
        else:
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
    def __init__(self, args: argparse.Namespace, split_provider: SplitProvider, split: str) -> None:
        """
        Args:
            args (argparse.Namespace): options
            split_provider (SplitProvider): Object of Splitprovider
            split (str): split
        """
        super().__init__()

        self.args = args
        self.split_provider = split_provider
        self.split = split

        self.df_source = self.split_provider.df_source
        self.df_split = self.df_source[self.df_source['split'] == self.split]

        self.raw_label_list = self.split_provider.raw_label_list
        self.internal_label_list = self.split_provider.internal_label_list
        self.input_list = self.split_provider.input_list

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


def create_dataloader(args: argparse.Namespace, split_provider: SplitProvider, split: str = None) -> DataLoader:
    """
    Creeate data loader ofr split.

    Args:
        args (argparse.Namespace): options
        split_provider (SplitProvider): Object of SplitProvider
        split (str, optional): split. Defaults to None.

    Returns:
        DataLoader: data loader
    """
    split_data = LoadDataSet(args, split_provider, split)

    # args never has both 'batch_size' and 'test_batch_size'.
    if args.isTrain:
        batch_size = args.batch_size
    else:
        batch_size = args.test_batch_size

    if args.sampler == 'yes':
        assert ((args.task == 'classification') or (args.task == 'deepsurv')), 'Cannot make sampler in regression.'
        assert (len(split_provider.raw_label_list) == 1), 'Cannot make sampler for multi-label.'
        shuffle = False
        sampler = _make_sampler(split_data)
    else:
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


def print_dataset_info(dataloaders: Dict[str, LoadDataSet]) -> None:
    """
    Print dataset size for each split.

    Args:
        dataloaders (Dict[str, DataLoader]): dictionary of split and its dataset
    """
    for split, dataloader in dataloaders.items():
        total = len(dataloader.dataset)
        logger.logger.info(f"{split:>5}_data = {total}")
    logger.logger.info('')

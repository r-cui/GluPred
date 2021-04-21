import datetime
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OhioDataset(Dataset):
    """ The OhioT1DM dataset for Torch training.
    """

    def __init__(self, raw_df, example_len, external_mean=None, external_std=None, unimodal=False):
        """
        Args
            raw_df: dataframe
            example_len: int
            external_mean: [float]
                If none, self fit.
            external_std: [float]
                If none, self fit.
            unimodal: bool
                If True, data contains glucose only
        """
        raw_df.replace(to_replace=-1, value=np.nan, inplace=True)
        self.example_len = example_len
        self.unimodal = unimodal
        self.data = self._initial(raw_df)  # (len, n_features)
        self.example_indices = self._example_indices()
        self._standardise(external_mean, external_std)
        print("Dataset loaded, total examples: {}.".format(len(self)))

        # post check
        for i in range(len(self)):
            if torch.isnan(self[i]).any():
                raise ValueError("NaN detected in dataset!")

    @staticmethod
    def str2dt(s):
        return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

    def _initial(self, raw_df):
        times = [self.str2dt(s) for s in raw_df["index_new"]]
        glucose = raw_df["glucose"].to_numpy(dtype=np.float32)
        basal = raw_df["basal"].to_numpy(dtype=np.float32)
        bolus = raw_df["bolus"].to_numpy(dtype=np.float32)
        bolus_dur = raw_df["bolus_dur"].to_numpy(dtype=np.float32)
        carbs = raw_df["carbs"].to_numpy(dtype=np.float32)

        bolus[np.isnan(bolus)] = 0.0
        carbs[np.isnan(carbs)] = 0.0

        total_len = len(times)

        # smooth out the long acting insulin
        i = 0
        while i < total_len:
            if bolus[i] > 0 and bolus_dur[i] > 0:
                # found a non-instant bolus
                j = 1
                while i + j < total_len:
                    if bolus[i + j] == bolus[i]:
                        j += 1
                    else:
                        break
                bolus[i: i + j] = bolus[i: i + j] / j
                i += j
            else:
                i += 1

        if self.unimodal:
            return np.array([
                glucose
            ], dtype=np.float32).T
        else:
            return np.array([
                glucose,
                basal,
                bolus,
                carbs,
            ], dtype=np.float32).T

    def _example_indices(self):
        """ Extract every possible example from the dataset, st. all data entry in this example is not missing.

        Returns:
            [(start_row, end_row)]
                Starting and ending indices for each possible example from this dataframe.
        """
        res = []
        total_len = self.data.shape[0]

        def look_ahead(start):
            end = start
            res = []
            while end < total_len:
                if np.any(np.isnan(self.data[end, :])):
                    break
                if end - start + 1 >= self.example_len:
                    res.append((end - self.example_len + 1, end))
                end += 1
            return res, end

        i = 0
        while i < total_len:
            if not np.any(np.isnan(self.data[i, :])):
                temp_res, temp_end = look_ahead(i)
                res += temp_res
                i = temp_end + 1
            else:
                i += 1
        return res

    def _standardise(self, external_mean=None, external_std=None):
        if external_mean is None and external_std is None:
            mean = []
            std = []
            for i in range(self.data.shape[1]):
                mean.append(np.nanmean(self.data[:, i]))
                std.append(np.nanstd(self.data[:, i]))
        else:
            mean = external_mean
            std = external_std
        self.mean = mean
        self.std = std
        for i in range(self.data.shape[1]):
            self.data[:, i] = (self.data[:, i] - mean[i]) / std[i]

    def __len__(self):
        return len(self.example_indices)

    def __getitem__(self, idx):
        """
        Args:
            idx: int
        Returns:
            (example_len, channels)
        """
        start_row, end_row = self.example_indices[idx]
        res = torch.from_numpy(self.data[start_row: end_row + 1, :])
        return res


def prepare_personal_data(train_csv_path, test_csv_path, example_len, unimodal):
    """ Prepare datasets for one patient.
    Args
        train_csv_path: str
        test_csv_path: str
        example_len: int
        unimodal: bool
    """
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)

    train_dataset = OhioDataset(train_df, example_len, unimodal=unimodal)
    test_dataset = OhioDataset(test_df, example_len, train_dataset.mean, train_dataset.std, unimodal=unimodal)

    return train_dataset, test_dataset


def prepare_global_data(data_dir, patient_id, example_len, unimodal):
    """ Prepare dataset for transfer learning.
    Args
        data_dir: str
        patient_id: str
        example_len: int
        unimodal: bool
    """
    train_csv_path = os.path.join(data_dir, "{}_train.csv".format(patient_id))
    test_csv_path = os.path.join(data_dir, "{}_test.csv".format(patient_id))
    train_dataset, test_dataset = prepare_personal_data(train_csv_path, test_csv_path, example_len, unimodal)

    patients = ["540", "544", "552", "559", "563", "567", "570", "575", "584", "588", "591", "596"]
    patients.remove(patient_id)
    global_training_set = torch.utils.data.ConcatDataset(
        [
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, "{}_train.csv".format(p))),
                example_len=example_len,
                external_mean=train_dataset.mean,
                external_std=train_dataset.std,
                unimodal=unimodal
            ) for p in patients
        ] + [
            OhioDataset(
                raw_df=pd.read_csv(os.path.join(data_dir, "{}_test.csv".format(p))),
                example_len=example_len,
                external_mean=train_dataset.mean,
                external_std=train_dataset.std,
                unimodal=unimodal
            ) for p in patients
        ]
    )
    return global_training_set, train_dataset, test_dataset
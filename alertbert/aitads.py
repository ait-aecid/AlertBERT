import datetime as dt
import json
from collections.abc import Iterator, Sequence
from typing import Any, Literal, Optional

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler

"""This module provides PyTorch datasets and samplers for the [AIT Alert Dataset](https://zenodo.org/records/8263181)."""

# base alert dataset classes


class AlertDataset(Dataset):
    """A custom dataset class for loading alert data.
    The input data is expected to be JSON files containing a list of JSON-objects.
    If multiple paths are provided, the data is loaded from all files, concatenated, and sorted by timestamp.
    For each path a time offset can be provided to adjust the timestamps.
    Furthermore, attack identifiers can be provided to create the hierarchical labels of AIT-ADS-A.

    Args:
        paths (str | Sequence[str]): The paths to the data files.
        time_offset (int | Sequence[int], optional): The time offset to apply to the data. Defaults to 0.
        attack_ids (Sequence[int], optional): The attack identifiers to use for the hierarchical labels of AIT-ADS-A. Defaults to None.
        time_keys (Sequence[str], optional): The keys in the data dictionaries that represent timestamps. Defaults to ["time", "raw_time"].
        time_sort_key (str, optional): The key to sort the data by timestamp. Defaults to "raw_time".

    Attributes:
        data (dict[str, np.array[Any]]): The loaded data.
        keys (dict_keys[str]): The keys of the data dictionaries.

    Methods:
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx: int | slice | np.ndarray[int]) -> dict[str, Any] | dict[str, Sequence]:
            Returns the data at the given index or slice. Supports cyclic indexing.
        __iter__(self) -> Iterator[dict[str, Any]]: Returns an iterator over the data.

    """

    def __init__(
        self,
        paths: str | Sequence[str],
        time_offset: int | Sequence[int] = 0,
        attack_ids: Sequence[int] = None,
        time_keys: Sequence[str] = ["time", "raw_time"],
        time_sort_key: str = "raw_time",
    ) -> None:
        super().__init__()
        paths = [paths] if isinstance(paths, str) else paths
        time_offset = (
            [time_offset] if isinstance(time_offset, int) else time_offset
        )
        assert len(paths) == len(time_offset), (
            "Number of paths and time offsets must match."
        )

        self.data = {}
        self.keys = []

        # load data from all paths
        for path, offset, att_id in zip(paths, time_offset, attack_ids):
            with open(path) as f:
                data = np.array(json.load(f))

            if self.keys:
                assert self.keys == list(data[0].keys()), "Keys of all data must match."
            else:
                self.keys = list(data[0].keys())

            # transpose data for faster access
            data = {k: np.array([d[k] for d in data]) for k in self.keys}

            if att_id:
                att_id = str(att_id)
                data["hierarchical_event_label"] = np.vectorize(
                    lambda x, a=att_id: x + a
                )(data["hierarchical_event_label"])

            for k in self.keys:
                if k in time_keys:
                    data[k] += offset

                if k in self.data:
                    self.data[k].append(data[k])
                else:
                    self.data[k] = [data[k]]
            
        # merge data and sort by timestamp
        self.data = {k: np.concatenate(v) for k, v in self.data.items()}
        if len(paths) > 1:
            sort_idx = np.argsort(self.data[time_sort_key])
            self.data = {k: v[sort_idx] for k, v in self.data.items()}

    def __len__(self) -> int:
        return len(self.data[self.keys[0]])

    def __getitem__(
        self, idx: int | slice | np.ndarray[int]
    ) -> dict[str, Any] | dict[str, np.ndarray[Any]]:
        if isinstance(idx, int):
            try:
                return {k: self.data[k][idx] for k in self.keys}
            except IndexError:
                idx %= len(self)
                return {k: self.data[k][idx] for k in self.keys}
        elif isinstance(idx, slice):
            start = idx.start if idx.start else 0
            stop = idx.stop if idx.stop else len(self)
            step = idx.step if idx.step else 1
            idx_array = np.arange(start=start, stop=stop, step=step)
        elif isinstance(idx, np.ndarray):
            idx_array = idx
        else:
            raise TypeError("Index must be an integer, slice or numpy array.")
        if idx_array[0] < 0 or idx_array[-1] >= len(self):
            idx_array = idx_array % len(self)
        return {k: self.data[k][idx_array] for k in self.keys}

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for i in range(len(self)):
            yield {k: self.data[k][i] for k in self.keys}


class MultiAlertDataset(Dataset):
    """A custom dataset class for loading multiple alert datasets.

    Args:
        path (str): The path to the dataset directory.
        split (Sequence[str] | Sequence[Sequence[str]]): The datasets to use.
        time_offsets (Sequence[Sequence[int]], optional): The time offset to apply to the data. Defaults to 0.
        attack_ids (Sequence[Sequence[int]], optional): The attack identifiers to use for the hierarchical labels of AIT-ADS-A. Defaults to None.

    Attributes:
        split (Sequence[str]): The datasets to use.
        path (str): The path to the dataset directory.
        keys (dict_keys[str]): The keys of the data dictionaries.
        scenarios (list[AlertDataset]): The individual subdatasets.
        n_scenarios (int): The number of subdatasets.

    Methods:
        __len__(self) -> int: Returns the length of the dataset.
        __getitem__(self, idx: tuple[int, int | slice]) ->
            dict[str, Any] | dict[str, Sequence]: Returns the data at the given index
            or slice. Supports cyclic indexing. Indices are are tuples of the form
            (scenario_index, data_index).
        __iter__(self) -> Iterator[dict[str, Any]]: Returns an iterator over the data.

    """

    def __init__(
        self,
        path: str,
        split: Sequence[str] | Sequence[Sequence[str]],
        time_offsets: Sequence[Sequence[int]] = None,
        attack_ids: Sequence[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if isinstance(split[0], str):
            self.split = [[s] for s in split]
        else:
            self.split = split

        if time_offsets is None:
            time_offsets = [[0] * len(s) for s in self.split]
        else:
            assert len(time_offsets) == len(self.split), (
                "Number of scenario time offsets must match number of scenarios."
            )
            assert all(len(to) == len(s) for to, s in zip(time_offsets, self.split)), (
                "Number of time offsets must match number of files."
            )
        
        if attack_ids is None:
            attack_ids = [[None] * len(s) for s in self.split]
        else:
            assert len(attack_ids) == len(self.split), (
                "Number of scenario attack ids must match number of scenarios."
            )
            assert all(len(att) == len(s) for att, s in zip(attack_ids, self.split)), (
                "Number of attack ids must match number of files."
            )

        self.n_scenarios = len(split)
        self.path = path

        self.scenarios = [
            AlertDataset(paths=[f"{path}/{file}" for file in scenario], time_offset=to, attack_ids=att_ids)
            for scenario, to, att_ids in zip(self.split, time_offsets, attack_ids)
        ]

        self.keys = self.scenarios[0].keys
        for scenario in self.scenarios[1:]:
            assert self.keys == scenario.keys, "Keys of all scenarios must match."

    def __getitem__(
        self, idx: tuple[int, int | slice | np.ndarray[int]]
    ) -> dict[str, Any] | dict[str, np.ndarray[Any]]:
        return self.scenarios[idx[0]][idx[1]]

    def __len__(self) -> int:
        return sum(list(len(s) for s in self.scenarios))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for scenario in self.scenarios:
            yield from scenario


# AIT Alert Dataset

## AIT Alert Dataset - Original

"""
scenario          | days | dirb | service_S | crack_pw | dns_S | online_C | split | cv split
---------------------------------------------------------------------------------------------
"fox"             |   4  | high |           |          |       |     yes  |  test |       1
"harrison"        |   4  | high |           |          |       |          |   val |       2
"russellmitchell" |   3  |  low |           |          |       |          |  test |       1
"santos"          |   3  |  low |           |          |       |     yes  |   val |       2
"shaw"            |   5  |  low |       no  |          |   yes |          | train |       3
"wardbeck"        |   4  |  low |           |          |       |          | train |       4
"wheeler"         |   4  | high |           |      no  |       |          | train |       3
"wilson"          |   5  | high |           |          |       |          | train |       4
"""

aitads_scenarios = [
    "fox",
    "harrison",
    "russellmitchell",
    "santos",
    "shaw",
    "wardbeck",
    "wheeler",
    "wilson",
]

aitads_train_val_test_split = {
    "train": ["shaw", "wardbeck", "wheeler", "wilson"],
    "val": ["harrison", "santos"],
    "test": ["fox", "russellmitchell"],
}

aitads_cv_split = {
    1: ["fox", "russellmitchell"],
    2: ["harrison", "santos"],
    3: ["shaw", "wheeler"],
    4: ["wardbeck", "wilson"],
}


aitads_train_external_mail_hosts = [
    "lane_mail",
    "wilsonahmed_mail",
    "butcherrowe_mail",
    "williams_mail",
    "smith_mail",
    "watkins_mail",
    "khanwood_mail",
    "fischer_mail",
    "sanchez_mail",
    "collinsbecker_mail",
    "walker_mail",
]

aitads_all_external_mail_hosts = [
    "lane_mail",
    "smith_mail",
    "wilsonahmed_mail",
    "butcherrowe_mail",
    "williams_mail",
    "jonesmorgan_mail",
    "hayes_mail",
    "watkins_mail",
    "khanwood_mail",
    "taylor_mail",
    "taylorcruz_mail",
    "rogersturnbull_mail",
    "davey_mail",
    "fischer_mail",
    "sanchez_mail",
    "whittaker_mail",
    "morris_mail",
    "miller_mail",
    "collinsbecker_mail",
    "walker_mail",
]


class AITAlertDatasetOriginal(MultiAlertDataset):
    """PyTorch dataset of the [AIT Alert Dataset](https://zenodo.org/records/8263181).

    Args:
        split (Literal["train", "val", "test", "all"] | list[str], optional): The split of the dataset to use. Defaults to "train".
        path (str, optional): The path to the data directory. Defaults to "alerts_json".
        drop_first_day (bool, optional): Whether to drop the first day of data for each scenario. Defaults to True.

    Attributes:
        split (Literal["train", "val", "test"] | list[str]): The split of the dataset to use.
        path (str): The path to the alerts directory.

    """

    def __init__(
        self,
        split: Literal["train", "val", "test", "all"] | list[str] = "train",
        path: str = "alerts_json",
        include_raw_data: bool = False,
        drop_first_day: bool = True,
    ) -> None:
        if split == "all":
            scenarios = aitads_scenarios
        elif isinstance(split, str):
            scenarios = aitads_train_val_test_split[split]
        else:
            scenarios = split

        if include_raw_data:
            scenarios = [f"{scenario}.json" for scenario in scenarios]
        else:
            scenarios = [f"{scenario}_light.json" for scenario in scenarios]

        super().__init__(split=scenarios, path=path)
        self.split = split

        if drop_first_day:
            for dataset in self.scenarios:
                first_day = dt.date.fromtimestamp(dataset[0]["time"])
                i = 0
                while dt.date.fromtimestamp(dataset[i]["time"]) == first_day:
                    i += 1
                dataset.data = dataset[i:]

        if include_raw_data:
            for i in self:
                i["raw_data"] = json.dumps(i["raw_data"], separators=(",", ":"))


## AIT Alert Dataset - Augmented


class AITAlertDatasetAugmented(MultiAlertDataset):
    """PyTorch dataset of the augmented AIT Alert Dataset (see the README in `../aitads_augmented` for more information).

    Args:
        split (Literal["train", "val", "test", "all"] | list[str], optional): The split of the dataset to use. Defaults to "train".
        configuration (str, optional): The configuration of the dataset. Defaults to "original".
        path (str, optional): The path to the data directory. Defaults to "aitads_augmented".

    Attributes:
        split (Literal["train", "val", "test"] | list[str]): The split of the dataset to use.
        configuration (str): The configuration of the dataset.
        path (str): The path to the alerts directory.

    """

    def __init__(
        self,
        split: Literal["train", "val", "test", "all"] | list[str] = "train",
        configuration: str = "original",
        path: str = "aitads_augmented",
    ) -> None:
        with open(f"{path}/configs/{configuration}.json") as f:
            self.config = json.load(f)

        if split == "all":
            scenario_templates = sum(
                (self.config[s] for s in ["train", "val", "test"]), start=[]
            )
        else:
            scenario_templates = self.config[split]


        # get filenames and start time offsets
        start_time = int(
            dt.datetime.fromisoformat(self.config["start_time"]).timestamp()
        )
        start_times_files = []
        files = []
        attack_ids = []
        attack_id = 1
        if split == "val":
            for scenario in self.config["train"]:
                for day in scenario:
                    attack_id += len(day["attacks"])
        elif split == "test":
            for scenario in self.config["train"] + self.config["val"]:
                for day in scenario:
                    attack_id += len(day["attacks"])

        for scenario in scenario_templates:
            scenario_files = []
            scenario_file_start_times = []
            scenario_attack_ids = []

            for i, day in enumerate(scenario):
                for file in day["noise"]:
                    scenario_files.append(file + ".json")
                    scenario_file_start_times.append(start_time + i * 86400)
                    scenario_attack_ids.append(0)
                for file, time in day["attacks"]:
                    scenario_files.append(file + ".json")
                    time = dt.time.fromisoformat(time)
                    scenario_file_start_times.append(
                        start_time
                        + i * 86400
                        + int(time.hour * 3600 + time.minute * 60 + time.second)
                    )
                    scenario_attack_ids.append(attack_id)
                    attack_id += 1

            files.append(scenario_files)
            start_times_files.append(scenario_file_start_times)
            attack_ids.append(scenario_attack_ids)

        super().__init__(
            path=path + "/data",
            split=files,
            time_offsets=start_times_files,
            attack_ids=attack_ids,
        )


## AIT Alert Dataset - Factory class


class AITAlertDataset:
    """Factory class for the AIT Alert Dataset. Returns either the original or augmented dataset depending on the flavour.
    See the respective dataset classes AITAlertDatasetOriginal and AITAlertDatasetAugmented for more information.

    Args:
        flavour (Literal["original", "augmented"], optional): The flavour of the dataset. Defaults to "original".
        **kwargs: Additional keyword arguments for the dataset.

    """

    def __new__(
        cls,
        flavour: Literal["original", "augmented"] = "augmented",
        **kwargs: dict,
    ) -> AITAlertDatasetOriginal | AITAlertDatasetAugmented:
        if flavour == "original":
            data = AITAlertDatasetOriginal(**kwargs)
        elif flavour == "augmented":
            data = AITAlertDatasetAugmented(**kwargs)
        else:
            raise ValueError(
                f"flavour must be either 'original' or 'augmented', got {flavour}"
            )
        data.flavour = flavour
        return data


# mnemonic alert dataset


class MnemonicAlertDataset(AlertDataset):
    """Dataset class for the Mnemonic Alert Dataset."""

    def __init__(self, path: str = "mnemonic_alerts") -> None:
        super().__init__(path + "/data.json")


# alert sequence samplers


class AlertSequenceSampler(Sampler):
    """A custom PyTorch sampler for sampling sequences from an alert dataset.

    Args:
        data_source (MultiAlertDataset | AlertDataset): The alert dataset to sample from.
        context_size (int, optional): The context window of each sampled sequence.
            Refers to number of tokens in a sequence if sampling_method is "index",
            if sampling_method is "time" then it refers to the length of a time interval to retreive in seconds.
            Defaults to 1024.
        sampling_method (Literal["index", "time"], optional): Whether to sample sequences evenly by index or timestamp.
        cyclic (bool, optional): Whether to sample sequences cyclically. Defaults to True. Ignored for time sampling.
        generator (Optional[torch.Generator], optional): The random number generator used for sampling. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the sequences. Defaults to True. Ignored for time sampling as in time sampling.

    Attributes:
        data_source (MultiAlertDataset | AlertDataset): The alert dataset to sample from.
        context_size (int): The size of each sampled sequence.
        index_sampling (bool): Whether to sample sequences by index.
        cyclic (bool): Whether to sample sequences cyclically.
        shuffle (bool): Whether to shuffle the sequences.
        generator (Optional[torch.Generator]): The random number generator.
        source_lengths (List[int]): The lengths of the subsets in the alert dataset.
        n_scenarios (int): The number of scenarios in the alert dataset.
        n_chunks (List[int]): The number of chunks in each scenario.
        margins (List[int]): The margins for each scenario.
        multi (bool): Whether the data source is a MultiAlertDataset.

    Methods:
        __iter__(self) -> Iterator[slice | tuple[int, slice]]: Returns an iterator over the sampled sequences.
            The sampled sequences have randomly chosen offsets.
        __len__(self) -> int: How many times context_size fits in the dataset.

    """

    def __init__(
        self,
        data_source: MultiAlertDataset | AlertDataset,
        context_size: int = 1024,
        sampling_method: Literal["index", "time"] = "index",
        cyclic: bool = True,
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.data_source = data_source
        self.context_size = context_size
        self.index_sampling = sampling_method == "index"
        self.cyclic = cyclic
        self.shuffle = shuffle if self.index_sampling else False
        self.generator = generator
        self.multi = isinstance(data_source, MultiAlertDataset)

        if not self.index_sampling:
            raise NotImplementedError(
                "Time sampling is not yet implemented. The existing code implemented a \
                deprecated version of time sampling which sampled fixed length \
                sequences of size `context_size` uniformly in the time of the scenario."
            )

        if self.multi:
            self.source_lengths = [len(d) for d in data_source.scenarios]
            self.n_scenarios = data_source.n_scenarios
            self.times = [
                (d.data["raw_time"][0], d.data["raw_time"][-1] - d.data["raw_time"][0])
                for d in data_source.scenarios
            ]
        else:
            self.source_lengths = [len(data_source)]
            self.n_scenarios = 1
            self.times = [
                (
                    data_source.data["raw_time"][0],
                    data_source.data["raw_time"][-1] - data_source.data["raw_time"][0],
                )
            ]

        # determine how many times context_size fits in each scenario
        self.n_chunks = [
            self.source_lengths[i] // self.context_size + self.cyclic
            for i in range(self.n_scenarios)
        ]
        self.margins = [
            self.source_lengths[i] % self.context_size for i in range(self.n_scenarios)
        ]

    def __iter__(self) -> Iterator[slice | tuple[int, slice]]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

        if self.index_sampling:
            # sample random offsets for each scenario
            # such that the chunks are different every epoch
            if self.cyclic:
                # we can use any offset
                # offsets are equivalent modulo context_size, hence we can sample
                # them uniformly from [0, context_size)
                offsets = torch.randint(
                    low=0,
                    high=self.context_size,
                    size=(self.n_scenarios,),
                    generator=self.generator,
                )
            else:
                # we need to sample offsets such that the chunks fit in the scenario
                # i.e. we need to sample from [0, margin)
                offsets = torch.tensor(
                    torch.randint(0, self.margins[i], generator=self.generator)
                    for i in range(self.n_scenarios)
                )

        # create a tensor containing the starting indices of the chunks
        indices = []
        for i in range(self.n_scenarios):
            chunk_indices = i * torch.ones((self.n_chunks[i], 2), dtype=torch.int64)
            if self.index_sampling:
                # sample uniformly from the scenario
                chunk_indices[:, 1] = (
                    torch.arange(self.n_chunks[i]) * self.context_size + offsets[i]
                )
            else:
                # sample uniformly from the time axis
                start_times = self.times[i][0] + self.times[i][1] * torch.rand(
                    (self.n_chunks[i],), generator=self.generator, dtype=torch.float64
                )
                chunk_indices[:, 1] = torch.searchsorted(
                    torch.tensor(
                        self.data_source.scenarios[i].data["raw_time"],
                        dtype=torch.float64,
                    ),
                    start_times,
                )
            indices.append(chunk_indices)
        indices = torch.cat(indices)

        if self.shuffle:
            perm = torch.randperm(len(self), generator=self.generator)
        else:
            perm = range(len(self))

        if self.multi:
            # yield scenario index and slice
            for i in perm:
                yield (
                    indices[i, 0].item(),
                    slice(
                        indices[i, 1].item(),
                        indices[i, 1].item() + self.context_size,
                        1,
                    ),
                )
        else:
            # yield slice only
            for i in perm:
                yield slice(
                    indices[i, 1].item(),
                    indices[i, 1].item() + self.context_size,
                    1,
                )

    def __len__(self) -> int:
        return sum(self.n_chunks)


class AlertSequenceBatchSampler(BatchSampler):
    """A batch sampler for generating batches of sequences from an alert dataset.

    Args:
        data_source (MultiAlertDataset | AlertDataset): The alert dataset to sample from.
        context_size (int): The size of the context window for each sequence. Defaults to 1024.
        sampling_method (Literal["index", "time"]): The method for sampling sequences. Defaults to "index".
            Sampling by "time" is only possible for batch_size=1 as neither padding nor support for nested tensors is implemented.
        cyclic (bool): Whether to sample sequences in a cyclic manner. Defaults to True.
        generator (torch.Generator | None): The random number generator used for sampling. Defaults to None.
        shuffle (bool): Whether to shuffle the sequences. Defaults to True.
        batch_size (int): The size of each batch. Defaults to 16.
        drop_last (bool): Whether to drop the last incomplete batch. Defaults to True.

    """

    def __init__(
        self,
        data_source: MultiAlertDataset | AlertDataset,
        context_size: int = 1024,
        sampling_method: Literal["index", "time"] = "index",
        cyclic: bool = True,
        generator: Optional[torch.Generator] = None,
        shuffle: bool = True,
        batch_size: int = 16,
        drop_last: bool = True,
    ) -> None:
        assert batch_size == 1 or sampling_method == "index", (
            "Sampling by time is only possible for batch_size=1 as neither padding nor \
                support for nested tensors is implemented."
        )
        super().__init__(
            AlertSequenceSampler(
                data_source, context_size, sampling_method, cyclic, generator, shuffle
            ),
            batch_size,
            drop_last,
        )

if __name__ == "__main__":
    val_data = AITAlertDataset(split="val", configuration="original")
    print(val_data.scenarios[0].data["hierarchical_event_label"][:1000])
    print(val_data.scenarios[1].data["hierarchical_event_label"][:1000])
# Copyright (c) Meta Platforms, Inc. and affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.idrid.dataset import IDRiDDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("idrid")
class IDRiDBuilder(MMFDatasetBuilder):
    def __init__(
        self, dataset_name="idrid", dataset_class=IDRiDDataset, *args, **kwargs
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/idrid/defaults.yaml"

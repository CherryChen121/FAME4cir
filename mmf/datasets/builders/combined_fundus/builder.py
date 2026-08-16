# Copyright (c) Meta Platforms, Inc. and affiliates.

from mmf.common.registry import registry
from mmf.datasets.builders.combined_fundus.dataset import CombinedFundusDataset
from mmf.datasets.mmf_dataset_builder import MMFDatasetBuilder


@registry.register_builder("combined_fundus")
class CombinedFundusBuilder(MMFDatasetBuilder):
    def __init__(
        self,
        dataset_name="combined_fundus",
        dataset_class=CombinedFundusDataset,
        *args,
        **kwargs,
    ):
        super().__init__(dataset_name, dataset_class, *args, **kwargs)

    @classmethod
    def config_path(cls):
        return "configs/datasets/combined_fundus/defaults.yaml"

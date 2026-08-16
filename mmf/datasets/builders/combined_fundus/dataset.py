# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from mmf.common.sample import Sample
from mmf.common.typings import MMFDatasetConfigType
from mmf.datasets.mmf_dataset import MMFDataset

from .database import CombinedFundusDatabase


class CombinedFundusDataset(MMFDataset):
    def __init__(
        self,
        config: MMFDatasetConfigType,
        dataset_type: str,
        index: int,
        *args,
        **kwargs,
    ):
        super().__init__(
            "combined_fundus",
            config,
            dataset_type,
            index,
            CombinedFundusDatabase,
            *args,
            **kwargs,
        )

    def init_processors(self):
        super().init_processors()
        if self._use_images:
            if self._dataset_type == "train":
                self.image_db.transform = self.train_image_processor
            else:
                self.image_db.transform = self.eval_image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        processed_sentence = self.text_processor(
            {"text": sample_info["sentences"]}
        )

        current_sample = Sample()
        current_sample.text = processed_sentence["text"]
        if "input_ids" in processed_sentence:
            current_sample.update(processed_sentence)

        image_extension = self.config.get("image_extension", ".jpg")
        if self._use_images:
            current_sample.ref_image = self.image_db.from_path(
                sample_info["ref_path"] + image_extension
            )["images"][0]
            current_sample.tar_image = self.image_db.from_path(
                sample_info["tar_path"] + image_extension
            )["images"][0]
        else:
            current_sample.ref_image = self.features_db.from_path(
                sample_info["ref_path"] + ".npy"
            )["image_feature_0"]
            current_sample.tar_image = self.features_db.from_path(
                sample_info["tar_path"] + ".npy"
            )["image_feature_0"]

        current_sample.ann_idx = torch.tensor(idx, dtype=torch.long)
        current_sample.targets = None
        current_sample.target_id = torch.tensor(
            sample_info["target_id"], dtype=torch.long
        )
        current_sample.fake_data = torch.tensor(
            sample_info["fake_data"], dtype=torch.bool
        )
        return current_sample

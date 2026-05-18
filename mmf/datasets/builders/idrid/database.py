# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import json

from mmf.utils.file_io import PathManager


class IDRiDDatabase:
    """Database for IDRiD CIR dataset (FashionIQ-compatible format).

    Expected directory structure:
        data_dir/
            captions/
                cap.IDRiD.train.json
                cap.IDRiD.val.json
            images/
                IDRiD/
                    IDRiD_001.jpg
                    ...

    Captions JSON format (same as FashionIQ):
        [{"candidate": "IDRiD/IDRiD_078", "target": "IDRiD/IDRiD_022", "captions": ["..."]}, ...]
    """

    SPLITS = {"train": "train", "val": "val", "test": "val"}

    def __init__(self, config, splits_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.splits = self.SPLITS[self.dataset_type]
        self._load_annotation_db(splits_path)

    def _load_annotation_db(self, splits_path):
        json_name = "cap.IDRiD.{}.json".format(self.splits)
        with PathManager.open(os.path.join(splits_path, json_name), "r") as f:
            annotations_json = json.load(f)

        self.data = []
        for item in annotations_json:
            self.data.append(
                {
                    "ref_path": item["candidate"],
                    "tar_path": item["target"],
                    "sentences": ", ".join(item["captions"]),
                }
            )

        if len(self.data) == 0:
            raise RuntimeError("Dataset is empty")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

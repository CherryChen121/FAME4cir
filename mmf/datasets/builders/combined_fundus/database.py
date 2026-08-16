# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import os

from mmf.utils.file_io import PathManager


class CombinedFundusDatabase:
    """FashionIQ-style annotations with an explicit full-image retrieval gallery."""

    INTERNAL_SUBSET = "Internal"
    TEST_SUBSETS = {"Internal", "GRAPE", "ODIR5K"}

    def __init__(self, config, captions_path, dataset_type, *args, **kwargs):
        super().__init__()
        self.dataset_type = dataset_type
        self.subset, self.split = self._resolve_split(config, dataset_type)
        self._load_annotation_db(captions_path)

    def _resolve_split(self, config, dataset_type):
        if dataset_type == "train":
            return self.INTERNAL_SUBSET, "train"
        if dataset_type == "val":
            return self.INTERNAL_SUBSET, "val"
        if dataset_type == "test":
            subset = config.get("test_subset", self.INTERNAL_SUBSET)
            if subset not in self.TEST_SUBSETS:
                choices = ", ".join(sorted(self.TEST_SUBSETS))
                raise ValueError(
                    f"Unknown combined fundus test_subset '{subset}'. "
                    f"Expected one of: {choices}"
                )
            return subset, "test"
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    def _load_json(self, path):
        if not PathManager.isfile(path):
            raise FileNotFoundError(path)
        with PathManager.open(path, "r") as stream:
            return json.load(stream)

    def _load_annotation_db(self, captions_path):
        dataset_root = os.path.dirname(captions_path.rstrip(os.sep))
        image_splits_path = os.path.join(dataset_root, "image_splits")
        captions_file = os.path.join(
            captions_path, f"cap.{self.subset}.{self.split}.json"
        )
        gallery_file = os.path.join(
            image_splits_path, f"split.{self.subset}.{self.split}.json"
        )

        annotations = self._load_json(captions_file)
        gallery_paths = self._load_json(gallery_file)
        if not annotations:
            raise RuntimeError(f"Dataset is empty: {captions_file}")
        if not gallery_paths:
            raise RuntimeError(f"Gallery is empty: {gallery_file}")
        if len(gallery_paths) != len(set(gallery_paths)):
            raise ValueError(f"Gallery contains duplicate image paths: {gallery_file}")

        gallery_ids = {path: index for index, path in enumerate(gallery_paths)}
        data = []
        query_targets = set()
        for item in annotations:
            candidate = item["candidate"]
            target = item["target"]
            if candidate not in gallery_ids:
                raise ValueError(
                    f"Candidate '{candidate}' is missing from {gallery_file}"
                )
            if target not in gallery_ids:
                raise ValueError(f"Target '{target}' is missing from {gallery_file}")
            captions = item.get("captions", [])
            if not captions:
                raise ValueError(
                    f"Query ({candidate}, {target}) has no modification caption"
                )
            data.append(
                {
                    "ref_path": candidate,
                    "tar_path": target,
                    "sentences": ", ".join(captions),
                    "target_id": gallery_ids[target],
                    "fake_data": False,
                }
            )
            query_targets.add(target)

        # Evaluation must rank against every image in image_splits, not only
        # targets appearing in the caption pairs. Dummy queries provide target
        # image features for the missing gallery entries and are filtered by the
        # metric through fake_data.
        if self.dataset_type != "train":
            for path in gallery_paths:
                if path not in query_targets:
                    data.append(
                        {
                            "ref_path": path,
                            "tar_path": path,
                            "sentences": "",
                            "target_id": gallery_ids[path],
                            "fake_data": True,
                        }
                    )

        self.data = data
        self.num_queries = len(annotations)
        self.gallery_size = len(gallery_paths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

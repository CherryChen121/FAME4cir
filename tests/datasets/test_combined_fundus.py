import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from mmf.datasets.builders.combined_fundus.database import CombinedFundusDatabase
from mmf.modules.metrics import RecallAtKCombinedFundus
from mmf.utils.configuration import load_yaml


class TestCombinedFundusDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dataset_root = self.temp_dir.name
        self.captions_dir = os.path.join(self.dataset_root, "captions")
        self.image_splits_dir = os.path.join(
            self.dataset_root, "image_splits"
        )
        os.makedirs(self.captions_dir)
        os.makedirs(self.image_splits_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_split(self, subset, split):
        captions = [
            {
                "candidate": "images/reference",
                "target": "images/target",
                "captions": ["show more lesions"],
            }
        ]
        gallery = [
            "images/reference",
            "images/target",
            "images/distractor",
        ]
        with open(
            os.path.join(self.captions_dir, f"cap.{subset}.{split}.json"),
            "w",
        ) as stream:
            json.dump(captions, stream)
        with open(
            os.path.join(
                self.image_splits_dir, f"split.{subset}.{split}.json"
            ),
            "w",
        ) as stream:
            json.dump(gallery, stream)

    def test_validation_adds_missing_gallery_images_as_fake_queries(self):
        self._write_split("Internal", "val")
        database = CombinedFundusDatabase(
            OmegaConf.create(), self.captions_dir, "val"
        )

        self.assertEqual(database.num_queries, 1)
        self.assertEqual(database.gallery_size, 3)
        self.assertEqual(len(database), 3)
        self.assertFalse(database[0]["fake_data"])
        self.assertTrue(database[1]["fake_data"])
        self.assertTrue(database[2]["fake_data"])
        self.assertEqual(
            {item["target_id"] for item in database}, {0, 1, 2}
        )

    def test_test_subset_selects_external_annotations(self):
        self._write_split("GRAPE", "test")
        database = CombinedFundusDatabase(
            OmegaConf.create({"test_subset": "GRAPE"}),
            self.captions_dir,
            "test",
        )

        self.assertEqual(database.subset, "GRAPE")
        self.assertEqual(database.split, "test")

    def test_rejects_unknown_test_subset(self):
        with self.assertRaisesRegex(ValueError, "Unknown combined fundus"):
            CombinedFundusDatabase(
                OmegaConf.create({"test_subset": "unknown"}),
                self.captions_dir,
                "test",
            )

    def test_recall_uses_fake_rows_only_to_complete_gallery(self):
        metric = RecallAtKCombinedFundus()
        target_ids = torch.tensor([0, 1, 2])
        fake_data = torch.tensor([False, False, True])
        embeddings = torch.eye(3)

        result = metric.calculate(
            {"target_id": target_ids, "fake_data": fake_data},
            {"comp_feats": embeddings, "tar_feats": embeddings},
        )

        self.assertEqual(result["R@1"].item(), 100.0)
        self.assertEqual(result["Avg_Recall"].item(), 100.0)


class TestCombinedFundusBackboneConfigs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config_dir = (
            Path(__file__).resolve().parents[2]
            / "projects"
            / "fashionclip"
            / "configs"
        )

    def _load_config(self, filename):
        return load_yaml(str(self.config_dir / filename))

    def test_base_and_large_configs_use_matching_local_assets(self):
        cases = (
            (
                "composition_combined_fundus.yaml",
                "clip-vit-base-patch16",
                32,
                "fashionclip_combined_fundus_composition_xattn",
            ),
            (
                "composition_combined_fundus_vitL14.yaml",
                "clip-vit-large-patch14",
                16,
                "fashionclip_combined_fundus_vitL14_composition_xattn",
            ),
        )

        for filename, backbone, batch_size, experiment_name in cases:
            with self.subTest(filename=filename):
                config = self._load_config(filename)
                model_path = config.model_config.fashionclip.clip_config.clip_model_name
                tokenizer_path = (
                    config.dataset_config.combined_fundus.processors
                    .text_processor.params.tokenizer_config.type
                )

                self.assertTrue(model_path.endswith(backbone))
                self.assertEqual(tokenizer_path, model_path)
                self.assertEqual(config.training.batch_size, batch_size)
                self.assertEqual(config.training.experiment_name, experiment_name)
                self.assertEqual(
                    config.evaluation.metrics,
                    ["r@k_combined_fundus"],
                )

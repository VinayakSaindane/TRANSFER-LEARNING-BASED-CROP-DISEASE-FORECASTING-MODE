"""Tests verifying dataset split integrity, zero leaf leakage, and PlantDoc separation."""

from __future__ import annotations

import csv
import json

from config.settings import DATASET_STATS_PATH, PLANTDOC_MANIFEST_PATH, SPLIT_MANIFEST_PATH


def test_dataset_statistics_and_zero_leakage() -> None:
    assert DATASET_STATS_PATH.exists(), "dataset_statistics.json must exist"
    assert SPLIT_MANIFEST_PATH.exists(), "splits.csv must exist"
    assert PLANTDOC_MANIFEST_PATH.exists(), "plantdoc_eval.csv must exist"

    stats = json.loads(DATASET_STATS_PATH.read_text(encoding="utf-8"))
    pv = stats["plantvillage"]
    pd_stats = stats["plantdoc"]

    assert pv["corrupted_removed_count"] == 0
    assert pv["leakage_verified_zero_overlap"] is True
    assert pd_stats["used_for_training"] is False

    splits_groups = {"train": set(), "val": set(), "test": set()}
    splits_hashes = {"train": set(), "val": set(), "test": set()}
    with SPLIT_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            splits_groups[row["split"]].add(row["group_id"])
            splits_hashes[row["split"]].add(row["sha256"])

    assert not (splits_groups["train"] & splits_groups["val"])
    assert not (splits_groups["train"] & splits_groups["test"])
    assert not (splits_groups["val"] & splits_groups["test"])

    assert not (splits_hashes["train"] & splits_hashes["val"])
    assert not (splits_hashes["train"] & splits_hashes["test"])
    assert not (splits_hashes["val"] & splits_hashes["test"])

    with PLANTDOC_MANIFEST_PATH.open("r", encoding="utf-8") as f:
        pd_hashes = {row["sha256"] for row in csv.DictReader(f)}
    all_pv_hashes = splits_hashes["train"] | splits_hashes["val"] | splits_hashes["test"]
    assert not (pd_hashes & all_pv_hashes)


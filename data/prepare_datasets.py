"""Dataset verification, leaf-grouped stratified splitting, and preprocessing pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.settings import (
    CLASS_NAMES,
    DATASET_STATS_PATH,
    IMAGE_SIZE,
    PLANTDOC_MANIFEST_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SPLIT_MANIFEST_PATH,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)

PLANTVILLAGE_DIR = RAW_DATA_DIR / "PlantVillage"
PLANTDOC_DIR = RAW_DATA_DIR / "PlantDoc"

FOLDER_SAFE_NAMES = {
    "Healthy": "Healthy",
    "Tomato Early Blight": "Tomato_Early_Blight",
}


def ensure_raw_datasets() -> None:
    """Download PlantVillage and PlantDoc subsets sparsely if not already present."""
    if not (PLANTVILLAGE_DIR / "raw" / "color" / "Tomato___healthy").exists():
        PLANTVILLAGE_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/spMohanty/PlantVillage-Dataset.git",
                str(PLANTVILLAGE_DIR),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(PLANTVILLAGE_DIR),
                "sparse-checkout",
                "set",
                "raw/color/Tomato___healthy",
                "raw/color/Tomato___Early_blight",
            ],
            check=True,
        )

    if not (PLANTDOC_DIR / "train" / "Tomato leaf").exists():
        PLANTDOC_DIR.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/pratikkayal/PlantDoc-Dataset.git",
                str(PLANTDOC_DIR),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(PLANTDOC_DIR),
                "sparse-checkout",
                "set",
                "train/Tomato leaf",
                "train/Tomato Early blight leaf",
                "test/Tomato leaf",
                "test/Tomato Early blight leaf",
            ],
            check=True,
        )


def _normalize_stem(filename: str) -> str:
    """Normalize PlantVillage image stem (stripping UUID prefix) to match leaf_grouping metadata."""
    name = filename.split("___", 1)[-1]
    stem = Path(name).stem
    stem = re.sub(r"_final_masked.*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"\s*\(.*?\)", "", stem)
    return " ".join(stem.lower().split())


def _fallback_leaf_group(label: str, filename: str) -> str:
    """Group multi-angle variants like 'GH_HL Leaf 259.1.JPG' and 'GH_HL Leaf 259.JPG' together."""
    norm = _normalize_stem(filename)
    m = re.match(r"^(gh_hl\s+leaf\s+\d+)(?:\.\d+)?$", norm)
    if m:
        return f"{label}::{m.group(1)}"
    return f"{label}::img_{norm}"


def load_plantvillage_leaf_groups() -> dict[tuple[str, str], str]:
    """Load physical leaf grouping IDs from PlantVillage filtered_leafmaps and leaf-map.json."""
    mapping: dict[tuple[str, str], str] = {}
    csv_files = {
        "Healthy": PLANTVILLAGE_DIR / "leaf_grouping" / "filtered_leafmaps" / "Tomato___healthy.csv",
        "Tomato Early Blight": PLANTVILLAGE_DIR / "leaf_grouping" / "filtered_leafmaps" / "Tomato___Early_blight.csv",
    }

    for label, csv_path in csv_files.items():
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = (row.get("File Name") or "").strip()
                leaf_num = (row.get("Leaf #") or "").strip()
                if fname and leaf_num:
                    mapping[(label, fname.lower())] = f"{label}::leaf_{leaf_num}"
                    mapping[(label, _normalize_stem(fname))] = f"{label}::leaf_{leaf_num}"

    leaf_map_json = PLANTVILLAGE_DIR / "leaf_grouping" / "leaf-map.json"
    if leaf_map_json.exists():
        try:
            raw_map = json.loads(leaf_map_json.read_text(encoding="utf-8"))
            for key, val in raw_map.items():
                if not val:
                    continue
                entry = val[0]
                if entry.startswith("Tomato___healthy:::"):
                    mapping.setdefault(("Healthy", _normalize_stem(key)), f"Healthy::{entry}")
                elif entry.startswith("Tomato___Early_blight:::"):
                    mapping.setdefault(("Tomato Early Blight", _normalize_stem(key)), f"Tomato Early Blight::{entry}")
        except (OSError, json.JSONDecodeError):
            pass

    return mapping


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_and_load_image(path: Path) -> tuple[bool, tuple[int, int] | None, str | None]:
    """Verify image integrity without relying on extension alone."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img2:
            rgb = img2.convert("RGB")
            size = rgb.size
        if size[0] < 32 or size[1] < 32:
            return False, size, "Image dimensions below 32x32 minimum"
        return True, size, None
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return False, None, str(exc)


def _save_resized_rgb(src_path: Path, dst_path: Path, size: tuple[int, int] = IMAGE_SIZE) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src_path) as img:
        rgb = img.convert("RGB")
        resized = rgb.resize(size, Image.Resampling.BILINEAR)
        resized.save(dst_path, format="JPEG", quality=95)


def grouped_stratified_split(
    records: list[dict[str, Any]],
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
    seed: int = RANDOM_SEED,
) -> dict[str, list[dict[str, Any]]]:
    """Assign entire leaf groups to train/val/test to guarantee zero leaf-level leakage."""
    rng = random.Random(seed)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    ratios = {"train": train_ratio, "val": val_ratio, "test": test_ratio}

    for label in CLASS_NAMES:
        cls_records = [r for r in records if r["label"] == label]
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in cls_records:
            groups[rec["group_id"]].append(rec)

        group_list = list(groups.items())
        rng.shuffle(group_list)
        # Assign larger groups first for closer target split proportions.
        group_list.sort(key=lambda kv: len(kv[1]), reverse=True)

        total_cls = len(cls_records)
        targets = {s: ratios[s] * total_cls for s in ("train", "val", "test")}
        counts = {"train": 0, "val": 0, "test": 0}

        for _, items in group_list:
            # Choose the split with the highest remaining deficit ratio.
            chosen = max(
                ("train", "val", "test"),
                key=lambda s: (targets[s] - counts[s]) / max(targets[s], 1.0),
            )
            splits[chosen].extend(items)
            counts[chosen] += len(items)

    return splits


def prepare_plantvillage(leaf_mapping: dict[tuple[str, str], str]) -> tuple[dict[str, Any], set[str]]:
    """Inspect, deduplicate, split, and resize PlantVillage tomato classes."""
    class_dirs = {
        "Healthy": PLANTVILLAGE_DIR / "raw" / "color" / "Tomato___healthy",
        "Tomato Early Blight": PLANTVILLAGE_DIR / "raw" / "color" / "Tomato___Early_blight",
    }

    raw_counts: dict[str, int] = {}
    corrupted_files: list[dict[str, str]] = []
    duplicate_files: list[dict[str, str]] = []
    seen_hashes: dict[str, str] = {}
    valid_records: list[dict[str, Any]] = []
    original_resolutions: Counter[str] = Counter()
    mapped_leaf_count = 0

    for label, src_dir in class_dirs.items():
        files = sorted(
            [p for p in src_dir.iterdir() if p.is_file() and not p.name.startswith(".")],
            key=lambda p: p.name,
        )
        raw_counts[label] = len(files)

        for img_path in files:
            ok, dims, err = _validate_and_load_image(img_path)
            if not ok:
                corrupted_files.append({"path": str(img_path), "label": label, "error": err or "invalid"})
                continue

            digest = _sha256_file(img_path)
            if digest in seen_hashes:
                duplicate_files.append(
                    {"path": str(img_path), "duplicate_of": seen_hashes[digest], "label": label}
                )
                continue
            seen_hashes[digest] = str(img_path)
            if dims:
                original_resolutions[f"{dims[0]}x{dims[1]}"] += 1

            gid = (
                leaf_mapping.get((label, img_path.name.split("___", 1)[-1].lower()))
                or leaf_mapping.get((label, _normalize_stem(img_path.name)))
            )
            if gid is not None:
                mapped_leaf_count += 1
            else:
                gid = _fallback_leaf_group(label, img_path.name)

            valid_records.append(
                {
                    "raw_path": img_path,
                    "filename": img_path.name,
                    "label": label,
                    "group_id": gid,
                    "sha256": digest,
                    "orig_width": dims[0] if dims else None,
                    "orig_height": dims[1] if dims else None,
                }
            )

    splits = grouped_stratified_split(valid_records)

    # Verify zero group leakage and zero hash leakage across splits.
    split_groups = {s: {r["group_id"] for r in items} for s, items in splits.items()}
    split_hashes = {s: {r["sha256"] for r in items} for s, items in splits.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap_g = split_groups[a] & split_groups[b]
        overlap_h = split_hashes[a] & split_hashes[b]
        if overlap_g or overlap_h:
            raise AssertionError(f"Data leakage detected between {a} and {b}: groups={overlap_g}, hashes={overlap_h}")

    # Write resized 224x224 RGB images and manifest CSV.
    for split_name in ("train", "val", "test"):
        for folder_name in FOLDER_SAFE_NAMES.values():
            target_dir = PROCESSED_DATA_DIR / split_name / folder_name
            if target_dir.exists():
                for old_f in target_dir.iterdir():
                    if old_f.is_file():
                        old_f.unlink()

    manifest_rows: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, int]] = {}
    for split_name, items in splits.items():
        cls_counter = Counter()
        for idx, rec in enumerate(sorted(items, key=lambda x: x["filename"])):
            safe_cls = FOLDER_SAFE_NAMES[rec["label"]]
            out_name = f"{split_name}_{idx:04d}_{Path(rec['filename']).stem}.jpg"
            out_path = PROCESSED_DATA_DIR / split_name / safe_cls / out_name
            _save_resized_rgb(rec["raw_path"], out_path, IMAGE_SIZE)
            cls_counter[rec["label"]] += 1
            manifest_rows.append(
                {
                    "split": split_name,
                    "label": rec["label"],
                    "group_id": rec["group_id"],
                    "sha256": rec["sha256"],
                    "processed_path": str(out_path.relative_to(BASE_DIR)),
                    "raw_path": str(rec["raw_path"].relative_to(BASE_DIR)),
                    "orig_width": rec["orig_width"],
                    "orig_height": rec["orig_height"],
                }
            )
        split_summary[split_name] = {
            "total": sum(cls_counter.values()),
            "Healthy": cls_counter["Healthy"],
            "Tomato Early Blight": cls_counter["Tomato Early Blight"],
        }

    with SPLIT_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "label",
                "group_id",
                "sha256",
                "processed_path",
                "raw_path",
                "orig_width",
                "orig_height",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    stats = {
        "source_name": "PlantVillage",
        "repository": "https://github.com/spMohanty/PlantVillage-Dataset",
        "license": "CC-BY-SA-3.0 (via Hugging Face dataset card metadata)",
        "raw_counts": raw_counts,
        "raw_total": sum(raw_counts.values()),
        "corrupted_removed_count": len(corrupted_files),
        "corrupted_files": corrupted_files,
        "exact_duplicates_removed_count": len(duplicate_files),
        "duplicate_files": duplicate_files,
        "valid_unique_total": len(valid_records),
        "unique_leaf_groups_total": len({r["group_id"] for r in valid_records}),
        "images_matched_to_leaf_map": mapped_leaf_count,
        "original_resolutions": dict(original_resolutions),
        "target_image_size": list(IMAGE_SIZE),
        "split_method": "Leaf-grouped stratified split (seed=42) preventing same-leaf leakage across splits",
        "splits": split_summary,
        "leakage_verified_zero_overlap": True,
    }
    return stats, set(seen_hashes.keys())


def prepare_plantdoc_external(pv_hashes: set[str]) -> dict[str, Any]:
    """Prepare PlantDoc tomato classes strictly as an isolated external field evaluation set."""
    sources = [
        ("Healthy", "train", PLANTDOC_DIR / "train" / "Tomato leaf"),
        ("Healthy", "test", PLANTDOC_DIR / "test" / "Tomato leaf"),
        ("Tomato Early Blight", "train", PLANTDOC_DIR / "train" / "Tomato Early blight leaf"),
        ("Tomato Early Blight", "test", PLANTDOC_DIR / "test" / "Tomato Early blight leaf"),
    ]

    for folder_name in FOLDER_SAFE_NAMES.values():
        tdir = PROCESSED_DATA_DIR / "plantdoc_eval" / folder_name
        if tdir.exists():
            for old_f in tdir.iterdir():
                if old_f.is_file():
                    old_f.unlink()

    raw_counts: Counter[str] = Counter()
    valid_counts: Counter[str] = Counter()
    corrupted: list[dict[str, str]] = []
    duplicates_internal = 0
    overlap_with_pv = 0
    seen_pd_hashes: set[str] = set()
    rows: list[dict[str, Any]] = []

    idx = 0
    for label, orig_partition, src_dir in sources:
        if not src_dir.exists():
            continue
        files = sorted(
            [p for p in src_dir.iterdir() if p.is_file() and not p.name.startswith(".")],
            key=lambda p: p.name,
        )
        raw_counts[label] += len(files)
        for img_path in files:
            ok, dims, err = _validate_and_load_image(img_path)
            if not ok:
                corrupted.append({"path": str(img_path), "label": label, "error": err or "invalid"})
                continue
            digest = _sha256_file(img_path)
            if digest in seen_pd_hashes:
                duplicates_internal += 1
                continue
            seen_pd_hashes.add(digest)
            if digest in pv_hashes:
                overlap_with_pv += 1
                continue

            safe_cls = FOLDER_SAFE_NAMES[label]
            out_name = f"pd_{orig_partition}_{idx:04d}.jpg"
            out_path = PROCESSED_DATA_DIR / "plantdoc_eval" / safe_cls / out_name
            _save_resized_rgb(img_path, out_path, IMAGE_SIZE)
            valid_counts[label] += 1
            rows.append(
                {
                    "source_partition": orig_partition,
                    "label": label,
                    "sha256": digest,
                    "processed_path": str(out_path.relative_to(BASE_DIR)),
                    "raw_path": str(img_path.relative_to(BASE_DIR)),
                    "orig_width": dims[0] if dims else None,
                    "orig_height": dims[1] if dims else None,
                }
            )
            idx += 1

    with PLANTDOC_MANIFEST_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_partition",
                "label",
                "sha256",
                "processed_path",
                "raw_path",
                "orig_width",
                "orig_height",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return {
        "source_name": "PlantDoc (Cropped Classification Subset)",
        "repository": "https://github.com/pratikkayal/PlantDoc-Dataset",
        "license": "CC-BY-4.0 (Attribution 4.0 International)",
        "used_for_training": False,
        "role": "External field-background domain-shift evaluation dataset only",
        "raw_counts": dict(raw_counts),
        "raw_total": sum(raw_counts.values()),
        "corrupted_removed_count": len(corrupted),
        "corrupted_files": corrupted,
        "internal_duplicates_removed": duplicates_internal,
        "overlap_with_plantvillage_hashes": overlap_with_pv,
        "valid_eval_counts": dict(valid_counts),
        "valid_eval_total": sum(valid_counts.values()),
    }


def main() -> None:
    ensure_raw_datasets()
    leaf_mapping = load_plantvillage_leaf_groups()
    pv_stats, pv_hashes = prepare_plantvillage(leaf_mapping)
    pd_stats = prepare_plantdoc_external(pv_hashes)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "classes": CLASS_NAMES,
        "plantvillage": pv_stats,
        "plantdoc": pd_stats,
    }

    DATASET_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DATASET_STATS_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

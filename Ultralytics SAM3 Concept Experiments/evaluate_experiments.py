#!/usr/bin/env python3
"""Unified confidence-free evaluator for SAM3 experiments on COCO and Cityscapes.

Metrics:
- per-class IoU
- per-class Dice
- mIoU
- mDice

Optional outputs:
- comparison grids that visualize Original / Ground Truth / SAM side by side
  for the selected classes.

This treats SAM as a labeler and evaluates mask overlap directly, without
using prediction confidences.
"""

from __future__ import annotations

import json
import hashlib
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools import mask as mask_utils
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pycocotools is required for evaluation. Install it with:\n"
        "  pip install pycocotools\n"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent

COCO_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
}

CITYSCAPES_CLASSES = {
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "traffic light",
    "traffic sign",
}


@dataclass(frozen=True)
class Prediction:
    class_name: str
    polygon: list[tuple[float, float]]


@dataclass(frozen=True)
class GroundTruthSample:
    key: str
    image_id: int
    image_path: Path
    width: int
    height: int
    payload: object | None = None


def load_dataset_names(dataset_yaml: Path) -> dict[int, str]:
    import yaml

    with open(dataset_yaml, "r") as f:
        data = yaml.safe_load(f) or {}

    names = data.get("names", {})
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def find_dataset_yaml(pred_dir: Path) -> Path:
    for parent in [pred_dir, *pred_dir.parents]:
        candidate = parent / "dataset.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find dataset.yaml for {pred_dir}")


def resolve_coco_image_path(ann_file: Path, file_name: str) -> Path:
    dataset_root = ann_file.parent.parent
    split_name = ann_file.stem
    if split_name.startswith("instances_"):
        split_name = split_name.removeprefix("instances_")

    candidates = [
        dataset_root / split_name / file_name,
        dataset_root / "images" / split_name / file_name,
        dataset_root / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    # Fall back to the most likely COCO layout even if the file is missing.
    return (dataset_root / split_name / file_name).resolve()


def parse_yolo_segmentation_label(label_path: Path, class_names: dict[int, str]) -> list[Prediction]:
    predictions: list[Prediction] = []
    if not label_path.exists():
        return predictions

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 7:
                continue

            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            polygon = list(zip(coords[0::2], coords[1::2]))
            predictions.append(Prediction(class_names[class_id], polygon))

    return predictions


def key_from_rel_path(rel_path: Path, cityscapes: bool = False) -> str:
    key = rel_path.with_suffix("").as_posix()
    if cityscapes:
        key = key.replace("_leftImg8bit", "").replace("_gtFine_polygons", "").replace("_gtFine_labelIds", "")
    return key


def polygon_to_mask(
    polygon: list[tuple[float, float]],
    width: int,
    height: int,
    normalized: bool,
) -> np.ndarray | None:
    if len(polygon) < 3:
        return None

    if normalized:
        pts = np.asarray(
            [[int(round(x * width)), int(round(y * height))] for x, y in polygon],
            dtype=np.int32,
        )
    else:
        pts = np.asarray([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)

    if pts.shape[0] < 3:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    return mask.astype(bool)


def load_prediction_groups(pred_dir: Path, class_names: dict[int, str], cityscapes: bool) -> dict[str, list[Prediction]]:
    groups: dict[str, list[Prediction]] = defaultdict(list)
    for label_path in tqdm(sorted(pred_dir.rglob("*.txt")), desc="Loading SAM predictions", unit="file"):
        rel = label_path.relative_to(pred_dir)
        preds = parse_yolo_segmentation_label(label_path, class_names)
        groups[key_from_rel_path(rel, cityscapes=cityscapes)].extend(preds)
    return groups


def select_classes(dataset_names: dict[int, str], supported: set[str], requested: list[str] | None) -> list[str]:
    dataset_classes = [name for _, name in sorted(dataset_names.items()) if name in supported]
    if requested is not None:
        return [cls for cls in requested if cls in supported and cls in dataset_names.values()]
    return dataset_classes


def build_coco_ground_truth(ann_file: Path, selected_classes: list[str]):
    coco_gt = COCO(str(ann_file))
    cat_lookup = {cat["name"]: int(cat["id"]) for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    cat_ids = {cls: cat_lookup[cls] for cls in selected_classes if cls in cat_lookup}

    samples: list[GroundTruthSample] = []
    for img in coco_gt.dataset["images"]:
        samples.append(
            GroundTruthSample(
                key=Path(img["file_name"]).stem,
                image_id=int(img["id"]),
                image_path=resolve_coco_image_path(ann_file, str(img["file_name"])),
                width=int(img["width"]),
                height=int(img["height"]),
                payload=None,
            )
        )

    return coco_gt, cat_ids, samples


def build_cityscapes_ground_truth(gt_root: Path, split: str, selected_classes: list[str]):
    gt_jsons = sorted((gt_root / "gtFine" / split).glob("*/*_gtFine_polygons.json"))
    images = []
    annotations = []
    samples: list[GroundTruthSample] = []
    cat_ids = {cls: idx for idx, cls in enumerate(selected_classes, start=1)}
    ann_id = 1
    image_id = 1

    for gt_json in tqdm(gt_jsons, desc="Loading Cityscapes GT", unit="file"):
        with open(gt_json, "r") as f:
            gt = json.load(f)

        width = int(gt["imgWidth"])
        height = int(gt["imgHeight"])
        key = key_from_rel_path(gt_json.relative_to(gt_root / "gtFine" / split), cityscapes=True)

        images.append(
            {
                "id": image_id,
                "file_name": str(gt_root / "leftImg8bit" / split / gt_json.parent.name / f"{gt_json.stem.replace('_gtFine_polygons', '')}_leftImg8bit.png"),
                "width": width,
                "height": height,
            }
        )
        image_path = gt_root / "leftImg8bit" / split / gt_json.parent.name / f"{gt_json.stem.replace('_gtFine_polygons', '')}_leftImg8bit.png"
        samples.append(
            GroundTruthSample(
                key=key,
                image_id=image_id,
                image_path=image_path.resolve(),
                width=width,
                height=height,
                payload=gt,
            )
        )

        for obj in gt.get("objects", []):
            label = obj.get("label")
            if label not in cat_ids:
                continue
            polygon = obj.get("polygon") or []
            if len(polygon) < 3:
                continue
            mask = polygon_to_mask([(float(x), float(y)) for x, y in polygon], width, height, normalized=False)
            if mask is None:
                continue
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("ascii")
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_ids[label],
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs) - min(xs)), float(max(ys) - min(ys))],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        image_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": idx, "name": cls, "supercategory": "cityscapes"} for cls, idx in cat_ids.items()],
    }
    coco_gt.createIndex()
    return coco_gt, cat_ids, samples


def gt_masks_for_sample_coco(coco_gt: COCO, sample: GroundTruthSample, selected_classes: list[str], cat_ids: dict[str, int]) -> dict[str, np.ndarray]:
    masks = {cls: np.zeros((sample.height, sample.width), dtype=bool) for cls in selected_classes}
    ann_ids = coco_gt.getAnnIds(imgIds=[sample.image_id])
    anns = coco_gt.loadAnns(ann_ids)
    for ann in anns:
        class_name = next((name for name, cid in cat_ids.items() if cid == int(ann["category_id"])), None)
        if class_name is None:
            continue
        masks[class_name] |= coco_gt.annToMask(ann).astype(bool)
    return masks


def gt_masks_for_sample_cityscapes(sample: GroundTruthSample, selected_classes: list[str]) -> dict[str, np.ndarray]:
    masks = {cls: np.zeros((sample.height, sample.width), dtype=bool) for cls in selected_classes}
    gt = sample.payload or {}
    for obj in gt.get("objects", []):
        label = obj.get("label")
        if label not in masks:
            continue
        polygon = obj.get("polygon") or []
        mask = polygon_to_mask([(float(x), float(y)) for x, y in polygon], sample.width, sample.height, normalized=False)
        if mask is None:
            continue
        masks[label] |= mask
    return masks


def pred_masks_for_sample(preds: list[Prediction], sample: GroundTruthSample, selected_classes: list[str]) -> dict[str, np.ndarray]:
    masks = {cls: np.zeros((sample.height, sample.width), dtype=bool) for cls in selected_classes}
    for pred in preds:
        if pred.class_name not in masks:
            continue
        mask = polygon_to_mask(pred.polygon, sample.width, sample.height, normalized=True)
        if mask is None:
            continue
        masks[pred.class_name] |= mask
    return masks


def class_color(class_name: str) -> tuple[int, int, int]:
    seed = int.from_bytes(hashlib.md5(class_name.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    color = rng.integers(64, 256, size=3)
    return int(color[0]), int(color[1]), int(color[2])


def load_bgr_image(image_path: Path, width: int, height: int) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height))
    return image


def overlay_class_masks(
    image: np.ndarray,
    masks: dict[str, np.ndarray],
    selected_classes: list[str],
    alpha: float = 0.45,
) -> np.ndarray:
    output = image.astype(np.float32).copy()
    for class_name in selected_classes:
        mask = masks.get(class_name)
        if mask is None or not mask.any():
            continue
        color = np.asarray(class_color(class_name), dtype=np.float32)
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def add_title_bar(image: np.ndarray, title: str, bar_height: int = 50) -> np.ndarray:
    framed = cv2.copyMakeBorder(image, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
    cv2.putText(
        framed,
        title,
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return framed


def create_comparison_grid(
    sample: GroundTruthSample,
    gt_masks: dict[str, np.ndarray],
    pred_masks: dict[str, np.ndarray],
    selected_classes: list[str],
) -> np.ndarray:
    original = load_bgr_image(sample.image_path, sample.width, sample.height)
    gt_overlay = overlay_class_masks(original, gt_masks, selected_classes)
    pred_overlay = overlay_class_masks(original, pred_masks, selected_classes)

    panels = [
        add_title_bar(original, "Original"),
        add_title_bar(gt_overlay, "Ground Truth"),
        add_title_bar(pred_overlay, "SAM"),
    ]
    return cv2.hconcat(panels)


def save_comparison_grid(
    sample: GroundTruthSample,
    gt_masks: dict[str, np.ndarray],
    pred_masks: dict[str, np.ndarray],
    selected_classes: list[str],
    grids_out: Path,
) -> None:
    grid = create_comparison_grid(sample, gt_masks, pred_masks, selected_classes)
    out_path = grids_out / f"{sample.key}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)


def accumulate_overlap_stats(gt_masks: dict[str, np.ndarray], pred_masks: dict[str, np.ndarray], stats: dict[str, dict[str, int]]) -> None:
    for class_name, gt_mask in gt_masks.items():
        pred_mask = pred_masks.get(class_name)
        if pred_mask is None:
            pred_mask = np.zeros_like(gt_mask)

        intersection = np.logical_and(gt_mask, pred_mask).sum()
        union = np.logical_or(gt_mask, pred_mask).sum()
        gt_sum = gt_mask.sum()
        pred_sum = pred_mask.sum()

        stats.setdefault(class_name, {"intersection": 0, "union": 0, "gt": 0, "pred": 0})
        stats[class_name]["intersection"] += int(intersection)
        stats[class_name]["union"] += int(union)
        stats[class_name]["gt"] += int(gt_sum)
        stats[class_name]["pred"] += int(pred_sum)


def evaluate_dataset(
    dataset_name: str,
    pred_dir: Path,
    selected_classes: list[str],
    out_file: Path | None,
    grids_out: Path | None = None,
    ann_file: Path | None = None,
    gt_root: Path | None = None,
    split: str = "val",
) -> dict:
    dataset_names = load_dataset_names(find_dataset_yaml(pred_dir))

    if dataset_name == "coco":
        supported = COCO_CLASSES
        if ann_file is None:
            raise ValueError("--ann_file is required for COCO evaluation")
        coco_gt, cat_ids, samples = build_coco_ground_truth(ann_file, selected_classes)
        predictions = load_prediction_groups(pred_dir, dataset_names, cityscapes=False)
        overlap_fn = lambda sample: gt_masks_for_sample_coco(coco_gt, sample, selected_classes, cat_ids)
    elif dataset_name == "cityscapes":
        supported = CITYSCAPES_CLASSES
        if gt_root is None:
            raise ValueError("--gt_root is required for Cityscapes evaluation")
        coco_gt, cat_ids, samples = build_cityscapes_ground_truth(gt_root, split, selected_classes)
        predictions = load_prediction_groups(pred_dir, dataset_names, cityscapes=True)
        overlap_fn = lambda sample: gt_masks_for_sample_cityscapes(sample, selected_classes)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    selected_classes = [cls for cls in selected_classes if cls in supported and cls in cat_ids]

    results = {
        "dataset": dataset_name,
        "prediction_dir": str(pred_dir),
        "classes": {},
        "summary": {},
        "prediction_counts": {},
    }
    if grids_out is not None:
        results["grids_out"] = str(grids_out)

    if not selected_classes:
        return results

    overlap_stats: dict[str, dict[str, int]] = {}
    sample_by_key = {sample.key: sample for sample in samples}
    for key, sample in tqdm(sample_by_key.items(), desc=f"Evaluating {dataset_name}", unit="img"):
        pred_masks = pred_masks_for_sample(predictions.get(key, []), sample, selected_classes)
        gt_masks = overlap_fn(sample)
        accumulate_overlap_stats(gt_masks, pred_masks, overlap_stats)
        if grids_out is not None:
            save_comparison_grid(sample, gt_masks, pred_masks, selected_classes, grids_out)

    prediction_counts: dict[str, int] = defaultdict(int)
    for pred_list in predictions.values():
        for pred in pred_list:
            if pred.class_name in selected_classes:
                prediction_counts[pred.class_name] += 1

    iou_values = []
    dice_values = []
    for class_name in selected_classes:
        s = overlap_stats.get(class_name, {"intersection": 0, "union": 0, "gt": 0, "pred": 0})
        intersection = s["intersection"]
        union = s["union"]
        gt_sum = s["gt"]
        pred_sum = s["pred"]
        iou = float(intersection / union) if union > 0 else float("nan")
        dice_den = gt_sum + pred_sum
        dice = float(2 * intersection / dice_den) if dice_den > 0 else float("nan")

        results["classes"][class_name] = {
            "iou": iou,
            "dice": dice,
            "intersection": intersection,
            "union": union,
            "gt": gt_sum,
            "pred": pred_sum,
        }

        if not np.isnan(iou):
            iou_values.append(iou)
        if not np.isnan(dice):
            dice_values.append(dice)

    results["summary"] = {
        "miou": float(np.mean(iou_values)) if iou_values else float("nan"),
        "mdice": float(np.mean(dice_values)) if dice_values else float("nan"),
        "num_classes": len(selected_classes),
        "num_predictions": int(sum(prediction_counts.values())),
    }
    results["prediction_counts"] = dict(prediction_counts)

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)

    return results


def print_report(results: dict) -> None:
    print("\nPer-class metrics")
    print(f"{'Class':<18} {'IoU':>10} {'Dice':>10}")
    print("-" * 42)
    for class_name, metrics in results["classes"].items():
        print(f"{class_name:<18} {metrics['iou']:10.4f} {metrics['dice']:10.4f}")

    summary = results["summary"]
    print("\nSummary")
    print(f"mIoU:          {summary['miou']:.4f}")
    print(f"mDice:         {summary['mdice']:.4f}")
    print(f"Classes:       {summary['num_classes']}")
    print(f"Predictions:   {summary['num_predictions']}")


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    coco = subparsers.add_parser("coco", help="Evaluate COCO predictions.")
    coco.add_argument("--ann_file", required=True, help="Path to instances_val2017.json or similar.")
    coco.add_argument("--pred_dir", required=True, help="Path to SAM3 labels directory.")
    coco.add_argument("--classes", nargs="*", default=None, help="Optional subset of class names.")
    coco.add_argument("--out", default=None, help="Optional JSON output path.")
    coco.add_argument("--grids_out", default=None, help="Optional output directory for comparison grids.")

    city = subparsers.add_parser("cityscapes", help="Evaluate Cityscapes predictions.")
    city.add_argument("--gt_root", required=True, help="Path to the Cityscapes dataset root.")
    city.add_argument("--split", default="val", help="Cityscapes split to evaluate against.")
    city.add_argument("--pred_dir", required=True, help="Path to SAM3 labels directory.")
    city.add_argument("--classes", nargs="*", default=None, help="Optional subset of class names.")
    city.add_argument("--out", default=None, help="Optional JSON output path.")
    city.add_argument("--grids_out", default=None, help="Optional output directory for comparison grids.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pred_dir = Path(args.pred_dir)
    out_file = Path(args.out) if getattr(args, "out", None) else None
    grids_out = Path(args.grids_out) if getattr(args, "grids_out", None) else None

    if args.dataset == "coco":
        dataset_names = load_dataset_names(find_dataset_yaml(pred_dir))
        selected = select_classes(dataset_names, COCO_CLASSES, args.classes)
        results = evaluate_dataset(
            "coco",
            pred_dir,
            selected,
            out_file,
            grids_out=grids_out,
            ann_file=Path(args.ann_file),
        )
    else:
        dataset_names = load_dataset_names(find_dataset_yaml(pred_dir))
        selected = select_classes(dataset_names, CITYSCAPES_CLASSES, args.classes)
        results = evaluate_dataset(
            "cityscapes",
            pred_dir,
            selected,
            out_file,
            grids_out=grids_out,
            gt_root=Path(args.gt_root),
            split=args.split,
        )

    print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

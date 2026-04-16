# Ultralytics SAM3 Concept Experiments

This folder contains a small workflow for running multiple SAM3 concept-mapping experiments and comparing their outputs side by side.

## Dataset layout

The scripts expect the datasets to already exist on disk with the following layouts.

### COCO

For inference, point `dataset.path` to the image split folder:

```text
datasets/
  coco/
    train2017/
    val2017/
    test2017/
    annotations/
      instances_train2017.json
      instances_val2017.json
      captions_train2017.json
      captions_val2017.json
```

Example config:

```yaml
dataset:
  path: /home/ruiz/Repositories/Utilities/Ultralytics SAM3 Concept Experiments/datasets/coco/val2017
```

For evaluation with `evaluate_experiments.py coco`, the annotation file should point to one of the COCO instance JSONs, such as:

```text
datasets/coco/annotations/instances_val2017.json
```

### Cityscapes

For inference, point `dataset.path` to the Cityscapes image split folder:

```text
datasets/
  cityscapes/
    leftImg8bit/
      train/
        <city>/*.png
      val/
        <city>/*.png
      test/
        <city>/*.png
    gtFine/
      train/
        <city>/*_gtFine_polygons.json
      val/
        <city>/*_gtFine_polygons.json
      test/
        <city>/*_gtFine_polygons.json
```

Example config:

```yaml
dataset:
  path: /home/ruiz/Repositories/Utilities/Ultralytics SAM3 Concept Experiments/datasets/cityscapes/leftImg8bit/val
```

For evaluation with `evaluate_experiments.py cityscapes`, the dataset root should be the folder that contains `leftImg8bit/` and `gtFine/`:

```text
datasets/cityscapes
```

### A2D2

If you use A2D2, keep the extracted dataset root intact and point `dataset.path` at the image root that contains the folders you want SAM to scan recursively.

## Run experiments

Example usage:
```bash
python sam3_concept_experiments.py --config configs/config_00_baseline.yaml
```

Arguments:
- `--model`: Path to the SAM3 model file. Defaults to `sam3.pt`.
- `--config`: Path to a single experiment config file. If omitted, all configs in `--configs_dir` are run.
- `--configs_dir`: Folder containing the experiment YAML files. Defaults to `configs/`.
- `--runs_dir`: Folder where experiment outputs will be written. Defaults to `runs/`.

What it does:

The script iterates over every `*.yaml` and `*.yml` file in the configs folder, runs the experiment for each one, and stores the results in a matching folder under `runs/`.
It now walks nested image folders recursively, so datasets like Cityscapes and A2D2 can be pointed at their extracted image roots directly.

Each experiment output includes:
- `images/train/`
- `labels/train/`
- `visualizations/`
- `dataset.yaml`

Important behavior:
- Visualizations are always generated.
- If an experiment already finished, it is skipped on the next run.
- Completion is tracked with a `.completed` marker file inside each experiment folder.

## Compare experiments

Example usage:
```bash
python generate_comparison_grids.py path/to/raw_images
```

Arguments:
- `raw_dir`: Path to the folder containing the raw images.
- `--runs_dir`: Base folder that contains experiment runs. Defaults to `runs/`.
- `--exps`: Optional list of experiment folder names or paths. If omitted, every run folder in `runs_dir` is used.
- `--out`: Output folder for comparison images. Defaults to `runs/comparisons`.

What it does:

The comparison script reads the original raw image plus each experiment's visualization for the same image path, then builds a horizontal grid for easy visual comparison.

## Evaluate Predictions

Install the evaluation dependency first:

```bash
pip install pycocotools
```

Unified usage:

```bash
python evaluate_experiments.py coco \
  --ann_file datasets/coco/annotations/instances_val2017.json \
  --pred_dir runs/coco_config/labels/train \
  --out runs/coco_config/coco_metrics.json \
  --grids_out runs/coco_config/comparison_grids

python evaluate_experiments.py cityscapes \
  --gt_root datasets/cityscapes \
  --split val \
  --pred_dir runs/cityscapes_config/labels/train \
  --out runs/cityscapes_config/cityscapes_metrics.json \
  --grids_out runs/cityscapes_config/comparison_grids
```

What it does:

- reads the SAM polygons from `runs/<exp>/labels/train/`
- maps class ids through `runs/<exp>/dataset.yaml`
- compares those masks to the dataset ground truth
- computes per-class IoU, per-class Dice, mIoU, and mDice
- optionally writes comparison grids with `Original / Ground Truth / SAM`
  side by side for the selected classes

## Config format

Each experiment config is a YAML file with a `concepts` mapping:

```yaml
concepts:
  person: pedestrian
  bicycle: bicycle
  car: car
  motorcycle: motorcycle
```

The left side is the final class name used in the dataset, and the right side is the prompt concept passed to SAM3.

## Output layout

```text
runs/
  config_00_baseline/
    .completed
    dataset.yaml
    images/
      train/
    labels/
      train/
    visualizations/
  comparisons/
```

## Notes

- Only `.png`, `.jpg`, and `.jpeg` files are processed.
- The experiment runner uses symbolic links for images when possible.
- Output labels and visualizations preserve the input folder structure, which avoids filename collisions in nested datasets.
- Comparison images are resized to match the original image size before being combined.

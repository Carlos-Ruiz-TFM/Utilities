# Ultralytics SAM3 Concept Experiments

This folder contains a small workflow for running multiple SAM3 concept-mapping experiments and comparing their outputs side by side.

## Run experiments

Example usage:
```bash
python sam3_concept_experiments.py path/to/raw_images
```

Arguments:
- `raw_path`: Path to the folder containing the raw images.
- `--model`: Path to the SAM3 model file. Defaults to `sam3.pt`.
- `--configs_dir`: Folder containing the experiment YAML files. Defaults to `configs/`.
- `--runs_dir`: Folder where experiment outputs will be written. Defaults to `runs/`.

What it does:

The script iterates over every `*.yaml` and `*.yml` file in the configs folder, runs the experiment for each one, and stores the results in a matching folder under `runs/`.

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
python compare_experiments.py path/to/raw_images
```

Arguments:
- `raw_dir`: Path to the folder containing the raw images.
- `--runs_dir`: Base folder that contains experiment runs. Defaults to `runs/`.
- `--exps`: Optional list of experiment folder names or paths. If omitted, every run folder in `runs_dir` is used.
- `--out`: Output folder for comparison images. Defaults to `runs/comparisons`.

What it does:

The comparison script reads the original raw image plus each experiment's visualization for the same filename, then builds a horizontal grid for easy visual comparison.

## Config format

Each experiment config is a YAML file with a `concepts` mapping:

```yaml
concepts:
  pedestrian: pedestrian
  bicycle: bicycle
  car: car
  motorcycle: motorcycle
```

The left side is the prompt concept passed to SAM3, and the right side is the final class name used in the dataset.

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
- Comparison images are resized to match the original image size before being combined.

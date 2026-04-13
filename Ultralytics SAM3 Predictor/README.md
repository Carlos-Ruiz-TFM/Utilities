# Ultralytics SAM3 Predictor

Example usage:
```bash
python sam3_predictor_queue.py path/to/raw_images --out path/to/output_dataset
```

Arguments:
- `raw_path`: Path to a folder containing the source images to process.
- `--out`: Output directory for the generated dataset. Defaults to `yolo_dataset`.

What it does:

This utility runs Ultralytics `SAM3SemanticPredictor` over a folder of images and converts the predictions into a YOLO-style segmentation dataset.

For each image, the script:
- loads the image into SAM3 with text prompts
- maps the predicted concept to one of the target dataset classes
- saves the image into `images/train`
- writes the corresponding polygon labels into `labels/train`
- generates a `dataset.yaml` file with the class names

Output structure:
```text
output_dataset/
  dataset.yaml
  images/
    train/
      *.jpg / *.png / *.jpeg
  labels/
    train/
      *.txt
```

Notes:
- The script expects a `sam3.pt` model file to be available when it runs.
- Only image files with `.png`, `.jpg`, or `.jpeg` extensions are processed.
- If no mask is produced for an image, no label file is written for that image.

Class mapping:
- `person` -> `pedestrian`
- `bicycle` -> `bicycle`
- `car`, `suv`, `van` -> `car`
- `motorcycle` -> `motorcycle`
- `bus` -> `bus`
- `heavy truck`, `trailer` -> `truck`
- `traffic light` -> `traffic light`
- `traffic sign` -> `traffic signs`
- `person riding a bicycle or electric scooter` -> `rider`
- `electric scooter` -> `electric scooter`
- `zebra crossing` -> `crosswalk`

Explanation:

The script uses a fixed prompt list to query SAM3 and then remaps the returned concept names into the final training classes. The saved labels are polygon coordinates normalized by the predictor, making them suitable for YOLO segmentation training workflows.

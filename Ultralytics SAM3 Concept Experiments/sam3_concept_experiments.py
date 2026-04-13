import os
import shutil
import threading
import yaml
from queue import Queue
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from ultralytics.models.sam import SAM3SemanticPredictor

CORRESPONDENCE = {
    "pedestrian": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
    "bus": 4, "truck": 5, "traffic light": 6, "traffic signs": 7,
    "rider": 8, "electric scooter": 9, "crosswalk": 10,
}

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS_DIR = SCRIPT_DIR / "configs"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
EXECUTION_MARKER = ".completed"
TARGET_CLASS_NAMES = {v: k for k, v in CORRESPONDENCE.items()}

class SAM3_Dataset_Predictor:
    def __init__(self, concept_mapping):
        self.predictor = None
        self.concept_mapping = concept_mapping
        self.prompts = list(concept_mapping.keys())
        self.write_queue = Queue()
        self.writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
        self.writer_thread.start()

    def _writer_worker(self):
        while True:
            item = self.write_queue.get()
            if item is None:
                self.write_queue.task_done()
                break
            output_path, content = item
            with open(output_path, "w") as f:
                f.write(content)
            self.write_queue.task_done()

    def load_predictor(self, model_path):
        overrides = dict(
            conf=0.4,
            task="segment",
            mode="predict",
            model=model_path,
            half=True,
            save=False,
            verbose=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def _generate_yaml(self, output_dataset_path):
        yaml_path = os.path.join(output_dataset_path, "dataset.yaml")
        yaml_data = {
            "train": "images/train",
            "val": "images/train",
            "nc": len(CORRESPONDENCE),
            "names": {v: k for k, v in CORRESPONDENCE.items()}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

    def _class_color(self, class_id):
        return (
            int(70 + (class_id * 53) % 170),
            int(90 + (class_id * 97) % 140),
            int(110 + (class_id * 29) % 120),
        )

    def _save_visualization(self, img_path, class_indices, polygons, vis_path):
        image = cv2.imread(img_path)
        if image is None:
            return

        overlay = image.copy()
        alpha = 0.35

        for c_idx, poly in zip(class_indices, polygons):
            if len(poly) == 0:
                continue

            concept = self.prompts[int(c_idx)]
            target_class = self.concept_mapping[concept]
            final_id = CORRESPONDENCE[target_class]
            label = TARGET_CLASS_NAMES[final_id]

            pts = np.array(
                [[int(pt[0] * image.shape[1]), int(pt[1] * image.shape[0])] for pt in poly],
                dtype=np.int32,
            )
            if len(pts) == 0:
                continue

            color = self._class_color(final_id)
            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=color, thickness=2)

            x, y, w, h = cv2.boundingRect(pts)
            text = label
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_y = max(y - 8, text_h + 8)
            cv2.rectangle(
                image,
                (x, text_y - text_h - baseline - 6),
                (x + text_w + 8, text_y + baseline),
                color,
                thickness=-1,
            )
            cv2.putText(
                image,
                text,
                (x + 4, text_y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        cv2.imwrite(vis_path, blended)

    def predict_dataset(self, raw_dataset_path, output_dataset_path):
        if self.predictor is None:
            raise ValueError("Predictor not loaded.")

        images_dir = os.path.join(output_dataset_path, "images", "train")
        labels_dir = os.path.join(output_dataset_path, "labels", "train")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        vis_dir = os.path.join(output_dataset_path, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)

        self._generate_yaml(output_dataset_path)
        valid_ext = (".png", ".jpg", ".jpeg")
        
        for img_name in tqdm(os.listdir(raw_dataset_path)):
            if not img_name.lower().endswith(valid_ext):
                continue
                
            img_path = os.path.join(raw_dataset_path, img_name)
            dest_img_path = os.path.join(images_dir, img_name)
            
            if not os.path.exists(dest_img_path):
                os.symlink(os.path.abspath(img_path), os.path.abspath(dest_img_path))

            self.predictor.set_image(img_path)
            results = self.predictor(text=self.prompts)
            
            if not results or results[0].masks is None:
                continue
                
            result = results[0]
            vis_path = os.path.join(vis_dir, img_name)

            class_indices = result.boxes.cls.cpu().tolist()
            polygons = result.masks.xyn
            self._save_visualization(img_path, class_indices, polygons, vis_path)

            output_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            lines = []

            for c_idx, poly in zip(class_indices, polygons):
                if len(poly) == 0: continue
                
                concept = self.prompts[int(c_idx)]
                target_class = self.concept_mapping[concept]
                final_id = CORRESPONDENCE[target_class]
                
                poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                lines.append(f"{final_id} {poly_str}\n")
                
            self.write_queue.put((output_path, "".join(lines)))

        self.write_queue.join()

    def cleanup(self):
        self.write_queue.put(None)
        self.writer_thread.join()


def load_concepts(config_path):
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}
    return config_data.get("concepts", {})


def iter_config_files(configs_dir):
    for config_path in sorted(Path(configs_dir).glob("*.y*ml")):
        if config_path.is_file():
            yield config_path


def run_experiment(raw_path, config_path, runs_dir, model_path):
    exp_name = config_path.stem
    output_path = Path(runs_dir) / exp_name
    marker_path = output_path / EXECUTION_MARKER

    if marker_path.exists():
        print(f"Skipping {exp_name}: already executed.")
        return

    concept_mapping = load_concepts(config_path)
    if not concept_mapping:
        print(f"Skipping {exp_name}: no concepts found in {config_path}.")
        return

    predictor = SAM3_Dataset_Predictor(concept_mapping)
    predictor.load_predictor(model_path)

    try:
        predictor.predict_dataset(raw_path, str(output_path))
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("completed\n")
        print(f"Finished {exp_name}.")
    finally:
        predictor.cleanup()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("raw_path", help="Path to the raw image folder.")
    parser.add_argument("--model", default="sam3.pt")
    parser.add_argument("--configs_dir", default=str(DEFAULT_CONFIGS_DIR), help="Directory containing experiment configs.")
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR), help="Directory where experiment outputs are stored.")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for config_path in iter_config_files(configs_dir):
        run_experiment(args.raw_path, config_path, runs_dir, args.model)

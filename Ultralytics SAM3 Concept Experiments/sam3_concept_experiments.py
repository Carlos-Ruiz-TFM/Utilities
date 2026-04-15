import os
import threading
import yaml
from queue import Queue
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from ultralytics.models.sam import SAM3SemanticPredictor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGS_DIR = SCRIPT_DIR / "configs"
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
EXECUTION_MARKER = ".completed"

ULTRALYTICS_COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199)
]
YOLO_COLORS_BGR = [(b, g, r) for r, g, b in ULTRALYTICS_COLORS]

class SAM3_Dataset_Predictor:
    def __init__(self, concept_mapping, overlap_threshold=0.6):
        self.predictor = None
        self.concept_mapping = concept_mapping
        self.prompts = list(concept_mapping.keys())
        
        self.target_classes = []
        for target in concept_mapping.values():
            if target not in self.target_classes:
                self.target_classes.append(target)
        self.correspondence = {cls: idx for idx, cls in enumerate(self.target_classes)}
        
        self.overlap_threshold = overlap_threshold
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
            imgsz=910,
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
            "nc": len(self.correspondence),
            "names": {v: k for k, v in self.correspondence.items()}
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, sort_keys=False)

    def _class_color(self, class_id):
        return YOLO_COLORS_BGR[class_id % len(YOLO_COLORS_BGR)]

    def _save_visualization(self, img_path, class_indices, polygons, vis_path):
        image = cv2.imread(img_path)
        if image is None:
            return

        overlay = np.zeros_like(image, dtype=np.uint8)
        alpha = 0.5 

        lw = max(round(sum(image.shape) / 2 * 0.003), 2)
        tf = max(lw - 1, 1)
        fs = lw / 3

        for c_idx, poly in zip(class_indices, polygons):
            if len(poly) == 0:
                continue

            concept = self.prompts[int(c_idx)]
            target_class = self.concept_mapping[concept]
            final_id = self.correspondence[target_class]
            label = self.target_classes[final_id]
            color = self._class_color(final_id)

            pts = np.array(
                [[int(pt[0] * image.shape[1]), int(pt[1] * image.shape[0])] for pt in poly],
                dtype=np.int32,
            )
            if len(pts) == 0:
                continue

            cv2.fillPoly(overlay, [pts], color)

            x, y, w, h = cv2.boundingRect(pts)
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

            w, h = cv2.getTextSize(label, 0, fontScale=fs, thickness=tf)[0]
            outside = p1[1] - h >= 3
            p2_bg = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2_bg, color, -1, cv2.LINE_AA)

            cv2.putText(
                image,
                label,
                (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                0,
                fs,
                (255, 255, 255), 
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

        mask = overlay.astype(bool)
        image[mask] = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)[mask]

        cv2.imwrite(vis_path, image)

    def _calculate_overlap_ratio(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0: return 0
        return intersection / union
        
    def _resolve_overlaps(self, class_indices, masks_data):
        num_masks = len(class_indices)
        if num_masks == 0: 
            return []

        # Ascending order: lowest index (0) has the highest priority
        sorted_detection_indices = np.argsort(-class_indices)
        
        keep_indices = []
        
        for i in sorted_detection_indices:
            mask_i = masks_data[i]
            discard = False
            
            for j in keep_indices:
                mask_j = masks_data[j]
                overlap_ratio = self._calculate_overlap_ratio(mask_i, mask_j)
                
                if overlap_ratio > self.overlap_threshold:
                    name_i = self.concept_mapping[self.prompts[class_indices[i]]]
                    name_j = self.concept_mapping[self.prompts[class_indices[j]]]
                    
                    print(f"Discarding mask {i} (class '{name_i}') due to overlap with mask {j} (class '{name_j}'), overlap ratio: {overlap_ratio:.2f}")                    
                    discard = True
                    break
                    
            if not discard:
                keep_indices.append(i)
                
        return keep_indices

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
            
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            masks_data = result.masks.data.cpu().numpy()
            polygons = result.masks.xyn
            
            keep_indices = self._resolve_overlaps(class_indices, masks_data)
            
            filtered_indices = [class_indices[idx] for idx in keep_indices]
            filtered_polygons = [polygons[idx] for idx in keep_indices]

            vis_path = os.path.join(vis_dir, img_name)
            self._save_visualization(img_path, filtered_indices, filtered_polygons, vis_path)

            output_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            lines = []

            for c_idx, poly in zip(filtered_indices, filtered_polygons):
                if len(poly) == 0: continue
                
                concept = self.prompts[int(c_idx)]
                target_class = self.concept_mapping[concept]
                final_id = self.correspondence[target_class]
                
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

def run_experiment(raw_path, config_path, runs_dir, model_path, overlap_threshold):
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

    predictor = SAM3_Dataset_Predictor(concept_mapping, overlap_threshold)
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
    parser.add_argument("--overlap_threshold", type=float, default=0.7, help="IoU threshold to resolve overlaps between related classes.")
    parser.add_argument("--configs_dir", default=str(DEFAULT_CONFIGS_DIR), help="Directory containing experiment configs.")
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR), help="Directory where experiment outputs are stored.")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    for config_path in iter_config_files(configs_dir):
        run_experiment(args.raw_path, config_path, runs_dir, args.model, args.overlap_threshold)
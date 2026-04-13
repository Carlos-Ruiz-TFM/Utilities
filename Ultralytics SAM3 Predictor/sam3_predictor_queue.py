import os
import shutil
import threading
import yaml
from queue import Queue
from tqdm import tqdm
from argparse import ArgumentParser
from ultralytics.models.sam import SAM3SemanticPredictor

CORRESPONDENCE = {
    "pedestrian": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "bus": 4,
    "truck": 5,
    "traffic light": 6,
    "traffic signs": 7,
    "rider": 8,
    "electric scooter": 9,
    "crosswalk": 10,
}

CONCEPT_MAPPING = {
    "person": "pedestrian",
    "bicycle": "bicycle",
    "car": "car",
    "suv": "car",
    "van": "car",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "heavy truck": "truck",
    "trailer": "truck",
    "traffic light": "traffic light",
    "traffic sign": "traffic signs",
    "person riding a bicycle or electric scooter": "rider",
    "electric scooter": "electric scooter",
    "zebra crossing": "crosswalk",
}

class SAM3_Dataset_Predictor:
    def __init__(self):
        self.predictor = None
        self.prompts = list(CONCEPT_MAPPING.keys())
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
        overrides = dict(conf=0.4, task="segment", mode="predict", model=model_path, half=True)
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

    def predict_dataset(self, raw_dataset_path, output_dataset_path):
        if self.predictor is None:
            raise ValueError("Predictor not loaded.")

        images_dir = os.path.join(output_dataset_path, "images", "train")
        labels_dir = os.path.join(output_dataset_path, "labels", "train")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        self._generate_yaml(output_dataset_path)
        valid_ext = (".png", ".jpg", ".jpeg")
        
        for img_name in tqdm(os.listdir(raw_dataset_path)):
            if not img_name.lower().endswith(valid_ext):
                continue
                
            img_path = os.path.join(raw_dataset_path, img_name)
            dest_img_path = os.path.join(images_dir, img_name)
            if not os.path.exists(dest_img_path):
                shutil.copy2(img_path, dest_img_path)

            self.predictor.set_image(img_path)
            results = self.predictor(text=self.prompts)
            
            if not results or results[0].masks is None:
                continue
                
            result = results[0]
            class_indices = result.boxes.cls.cpu().tolist()
            polygons = result.masks.xyn

            output_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")
            lines = []

            for c_idx, poly in zip(class_indices, polygons):
                if len(poly) == 0: continue
                
                # Double mapping logic
                concept = self.prompts[int(c_idx)]
                target_class = CONCEPT_MAPPING[concept]
                final_id = CORRESPONDENCE[target_class]
                
                poly_str = " ".join([f"{pt[0]:.6f} {pt[1]:.6f}" for pt in poly])
                lines.append(f"{final_id} {poly_str}\n")
                
            self.write_queue.put((output_path, "".join(lines)))

        self.write_queue.join()

    def cleanup(self):
        self.write_queue.put(None)
        self.writer_thread.join()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("raw_path")
    parser.add_argument("--out", default="yolo_dataset")
    args = parser.parse_args()

    predictor = SAM3_Dataset_Predictor()
    predictor.load_predictor("sam3.pt")
    try:
        predictor.predict_dataset(args.raw_path, args.out)
    finally:
        predictor.cleanup()
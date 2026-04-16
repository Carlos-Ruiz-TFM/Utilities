import os
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUNS_DIR = SCRIPT_DIR / "runs"
VALID_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def resolve_experiment_dir(exp, runs_dir):
    exp_path = Path(exp)
    if exp_path.is_absolute():
        return exp_path
    return Path(runs_dir) / exp_path


def create_comparison_grid(raw_dir, exp_dirs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    raw_root = Path(raw_dir)
    images = [
        path for path in sorted(raw_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
    ]

    for img_path in tqdm(images, desc="Generating comparisons"):
        orig_img = cv2.imread(str(img_path))
        
        if orig_img is None:
            continue

        img_list = [orig_img]
        labels = ["Original"]
        h, w = orig_img.shape[:2]

        for exp in exp_dirs:
            rel_path = img_path.relative_to(raw_root)
            exp_vis_path = Path(exp) / "visualizations" / rel_path
            labels.append(os.path.basename(exp))
            
            if exp_vis_path.exists():
                exp_img = cv2.imread(str(exp_vis_path))
                exp_img = cv2.resize(exp_img, (w, h))
                img_list.append(exp_img)
            else:
                img_list.append(np.zeros_like(orig_img))

        bar_height = 50
        labeled_imgs = []
        
        for img, label in zip(img_list, labels):
            padded = cv2.copyMakeBorder(img, bar_height, 0, 0, 0, cv2.BORDER_CONSTANT, value=(40, 40, 40))
            cv2.putText(padded, label, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            labeled_imgs.append(padded)

        grid = cv2.hconcat(labeled_imgs)

        out_path = Path(output_dir) / img_path.relative_to(raw_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), grid)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("raw_dir", help="Path to the raw image folder.")
    parser.add_argument("--runs_dir", default=str(DEFAULT_RUNS_DIR), help="Base directory that contains experiment runs.")
    parser.add_argument("--exps", nargs="*", help="Experiment folder names or paths. Defaults to every run folder in runs_dir.")
    parser.add_argument("--out", default=None, help="Output directory for comparison images.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if args.exps:
        exp_dirs = [str(resolve_experiment_dir(exp, runs_dir)) for exp in args.exps]
    else:
        exp_dirs = [
            str(path)
            for path in sorted(runs_dir.iterdir())
            if path.is_dir() and path.name != "comparisons"
        ]

    if not exp_dirs:
        print(f"No experiment runs found in {runs_dir}.")
        raise SystemExit(0)

    out_dir = args.out or str(runs_dir / "comparisons")
    create_comparison_grid(args.raw_dir, exp_dirs, out_dir)

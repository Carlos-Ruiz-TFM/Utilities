"Extract non-redundant frames from a video based on a specified threshold."
import argparse
import cv2
import numpy as np
import os
import tqdm

def extract_unique_frames(input_path, output_dir, threshold=10.0) -> None:
    """Method to extract unique frames from a video or a directory of frames based on a specified threshold.

    Parameters
    ----------
    input_path : str
        Path to the input video file or directory of frames.
    output_dir : str
        Path to the output directory where unique frames will be saved.
    threshold : float, optional
        Threshold for frame difference (default: 10.0)
    """
    if os.path.isdir(input_path):
        folder_name = f"{input_path.split('/')[-1]}_non_redundant_frames_{threshold}"
    else:
        folder_name = f"{input_path.split('/')[-1].split('.')[0]}_non_redundant_frames_{threshold}"
    output_dir = os.path.join(output_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.isdir(input_path):
        frame_files = sorted([f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png'))])
        if not frame_files:
            print("No image files found in the provided directory.")
            return
        
        prev_frame = cv2.imread(os.path.join(input_path, frame_files[0]))
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, "frame_0000.jpg"), prev_frame)
        
        saved_count = 1
        
        for frame_file in tqdm.tqdm(frame_files[1:], desc="Processing frames"):
            frame = cv2.imread(os.path.join(input_path, frame_file))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                filename = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                prev_gray = gray
                saved_count += 1
    else:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error opening video file.")
            return
        
        ret, prev_frame = cap.read()
        if not ret:
            print("Error reading the first frame.")
            return
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_dir, "frame_0000.jpg"), prev_frame)
        
        saved_count = 1
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_gray, gray)
            mean_diff = np.mean(diff)
            
            if mean_diff > threshold:
                filename = f"frame_{saved_count:04d}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), frame)
                prev_gray = gray
                saved_count += 1
        
        cap.release()
    print(f"Extracted {saved_count} unique frames to {output_dir}.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Extract unique frames from a video based on a threshold.")
    argparser.add_argument("input_path", help="Path to the input video file or directory of frames. Frames inside the directory are required to be ordered starting from 0. E.g 000.jpg, 001.jpg, 002.jpg, ...")
    argparser.add_argument("--output_dir", default=None, help="Path to the output directory where a folder of unique frames will be saved. If no path is provided, it will be saved in the current directory.")
    argparser.add_argument("--threshold", type=float, default=10.0, help="Threshold for frame difference (default: 10.0)")
    
    if not argparser.parse_args().output_dir:
        output_dir = os.getcwd()
    else:
        output_dir = argparser.parse_args().output_dir
    
    args = argparser.parse_args()
    extract_unique_frames(args.input_path, output_dir, threshold=args.threshold)

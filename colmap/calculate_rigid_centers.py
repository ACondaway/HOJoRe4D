import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name", required=True)
    parser.add_argument("--keypoints_file", type=str, default="keypoints.npy", help="name of the keypoints file")
    parser.add_argument("--output_file", type=str, default="rigid_centers.json", help="name of the output JSON file")
    return parser.parse_args()

def calculate_rigid_centers(keypoints_file, output_file):
    # Load keypoints from the npy file
    keypoints = np.load(keypoints_file)

    # Calculate the rigid center for each frame
    rigid_centers = []
    for points in keypoints:
        center = np.mean(points, axis=0)
        rigid_centers.append(center.tolist())

    # Save the rigid centers to a JSON file
    with open(output_file, 'w') as f:
        json.dump(rigid_centers, f)

    print(f"Rigid centers saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    keypoints_file_path = os.path.join("/home/sjtu/eccv_workspace/hold/code/data/", args.seq_name, "processed/colmap_2d", args.keypoints_file)
    output_file_path = os.path.join(args.seq_name, "processed/colmap_2d", args.output_file)
    calculate_rigid_centers(keypoints_file_path, output_file_path)

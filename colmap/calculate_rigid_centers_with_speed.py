import os
import json
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name", required=True)
    parser.add_argument("--keypoints_file", type=str, default="keypoints.npy", help="name of the keypoints file")
    parser.add_argument("--output_file", type=str, default="rigid_centers_with_speed.json", help="name of the output JSON file")
    parser.add_argument("--n_frames", type=int, default=5, help="number of previous frames to consider for velocity calculation")
    parser.add_argument("--weight_current", type=float, default=0.7, help="weight for the current center in the final adjustment")
    parser.add_argument("--weight_predicted", type=float, default=0.3, help="weight for the predicted center in the final adjustment")
    return parser.parse_args()

def calculate_rigid_centers(keypoints_file, output_file, n_frames, weight_current, weight_predicted):
    # Load keypoints from the npy file
    keypoints = np.load(keypoints_file)

    # Initialize list to store rigid centers
    rigid_centers = []

    for idx, points in enumerate(keypoints):
        # Calculate the current frame's rigid center
        current_center = np.mean(points, axis=0)
        if idx > 0:
            if len(rigid_centers) > 1:
                previous_centers = np.array(rigid_centers[max(0, idx-n_frames):idx])
                velocities = np.diff(previous_centers, axis=0)
                if velocities.size > 0:
                    average_velocity = np.mean(velocities, axis=0)
                else:
                    average_velocity = np.zeros_like(current_center)
            else:
                average_velocity = np.zeros_like(current_center)

            # Calculate the predicted center
            previous_center = np.array(rigid_centers[-1])
            predicted_center = previous_center + average_velocity

            # Calculate the adjusted center
            adjusted_center = (weight_current * current_center) + (weight_predicted * predicted_center)
        else:
            adjusted_center = current_center

        rigid_centers.append(adjusted_center.tolist())

    # Save the rigid centers to a JSON file
    with open(output_file, 'w') as f:
        json.dump(rigid_centers, f)

    print(f"Rigid centers saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    keypoints_file_path = os.path.join("/home/sjtu/eccv_workspace/hold/code/data/", args.seq_name, "processed/colmap_2d", args.keypoints_file)
    output_file_path = os.path.join("/home/sjtu/eccv_workspace/hold/code/data/", args.seq_name, "processed/colmap_2d", args.output_file)
    calculate_rigid_centers(keypoints_file_path, output_file_path, args.n_frames, args.weight_current, args.weight_predicted)

import os
import json
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name", required=True)
    parser.add_argument("--image_folder", type=str, help="path to the folder containing images", required=True)
    parser.add_argument("--center_file", type=str, default="rigid_centers.json", help="name of the JSON file with rigid centers")
    parser.add_argument("--output_folder", type=str, help="path to save images with rigid centers", required=True)
    return parser.parse_args()

def draw_rigid_centers(image_folder, center_file, output_folder):
    # Load rigid centers from JSON file
    with open(center_file, 'r') as f:
        rigid_centers = json.load(f)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the list of image files sorted by name
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for idx, image_file in enumerate(image_files):
        # Read the image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Failed to load image {image_path}")
            continue

        # Get the rigid center for the current image
        center = rigid_centers[idx]
        x_center, y_center = map(int, center)
        
        # Draw the rigid center on the image
        cv2.circle(image, (x_center, y_center), radius=5, color=(0, 0, 255), thickness=-1)
        
        # Save the image with the rigid center
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed {image_file} with center {center}")

if __name__ == "__main__":
    args = parse_args()
    center_file_path = os.path.join("/home/sjtu/eccv_workspace/hold/code/data/", args.seq_name, "processed/colmap_2d", args.center_file)
    draw_rigid_centers(args.image_folder, center_file_path, args.output_folder)

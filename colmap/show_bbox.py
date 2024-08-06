import os
import json
import cv2
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_name", type=str, help="sequence name", required=True)
    parser.add_argument("--image_folder", type=str, help="path to the folder containing images", required=True)
    parser.add_argument("--bbox_file", type=str, default="bboxes.json", help="name of the bbox JSON file")
    parser.add_argument("--output_folder", type=str, help="path to save images with bounding boxes", required=True)
    return parser.parse_args()

def draw_bboxes(image_folder, bbox_file, output_folder):
    # Load bounding boxes from JSON file
    with open(bbox_file, 'r') as f:
        bboxes = json.load(f)

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

        # Get the bounding box for the current image
        bbox = bboxes[idx]
        x_min, y_min, x_max, y_max = map(int, bbox)
        
        # Draw the bounding box on the image
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Save the image with bounding box
        output_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_path, image)

        print(f"Processed {image_file} with bbox {bbox}")

if __name__ == "__main__":
    args = parse_args()
    bbox_file_path = os.path.join(args.image_folder, args.bbox_file)
    draw_bboxes(args.image_folder, bbox_file_path, args.output_folder)

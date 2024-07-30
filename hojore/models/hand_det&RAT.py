from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full

from vitpose_model import ViTPoseModel
from rat_h import RelationAwareTokenization

def calculate_hand_bounding_boxes(j2d_r, j2d_l):
    def get_bbox_from_keypoints(keypoints):
        min_pt = np.min(keypoints, axis=0)
        max_pt = np.max(keypoints, axis=0)
        cx = (min_pt[0] + max_pt[0]) / 2
        cy = (min_pt[1] + max_pt[1]) / 2
        sx = max_pt[0] - min_pt[0]
        sy = max_pt[1] - min_pt[1]
        return np.array([cx, cy, sx, sy])

    bbox_r = get_bbox_from_keypoints(j2d_r)
    bbox_l = get_bbox_from_keypoints(j2d_l)
    return np.array([bbox_r, bbox_l])

def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Use the ViTPose model for keypoint detection
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Assuming a single person in the image for simplicity
    vitpose_results = cpm.predict_pose(image)
    if not vitpose_results:
        raise ValueError("No pose detected.")

    # Extract right and left hand keypoints
    keypoints = vitpose_results[0]['keypoints']
    j2d_right = keypoints[-21:]  # Right hand keypoints (assuming the model provides 21 keypoints per hand)
    j2d_left = keypoints[-42:-21]  # Left hand keypoints

    bounding_boxes = calculate_hand_bounding_boxes(j2d_right, j2d_left)

    img_size = image.shape[:2]
    patch_size = 16  # Example patch size; adjust as needed
    embed_dim = 64   # Example embedding dimension; adjust as needed
    rat_model = RelationAwareTokenization(img_size[0], patch_size, embed_dim)
    bounding_boxes_tensor = torch.tensor(bounding_boxes).float()

    patches = torch.zeros((1, patch_size, patch_size, 3))  # Placeholder for patches
    relation_aware_tokens = rat_model(patches, bounding_boxes_tensor)

    relative_distance_map = rat_model.calculate_relative_distance_map(
        rat_model.calculate_position_maps(bounding_boxes_tensor, img_size[0], img_size[1], patches.device)[:, 0, :, :],
        rat_model.calculate_position_maps(bounding_boxes_tensor, img_size[0], img_size[1], patches.device)[:, 1, :, :]
    )
    overlapping_map = rat_model.calculate_overlapping_map(bounding_boxes_tensor, img_size[0], img_size[1], patches.device)

    # Output the results
    output_image_with_bboxes(image, bounding_boxes)
    save_maps(relative_distance_map, overlapping_map)

def output_image_with_bboxes(image, bounding_boxes):
    for bbox in bounding_boxes:
        cx, cy, sx, sy = bbox
        start_point = (int(cx - sx / 2), int(cy - sy / 2))
        end_point = (int(cx + sx / 2), int(cy + sy / 2))
        cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)
    cv2.imwrite("output_image_with_bboxes.jpg", image)

def save_maps(relative_distance_map, overlapping_map):
    np.save("relative_distance_map.npy", relative_distance_map.cpu().numpy())
    np.save("overlapping_map.npy", overlapping_map.cpu().numpy())

if __name__ == "__main__":
    image_path = "input_image.jpg"  # Replace with your image path
    process_image(image_path)

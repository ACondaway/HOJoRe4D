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
from rat_h import RelativeAttentionTokenization, get_dis_tok, get_overlapping_map

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

    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    vitpose_results = cpm.predict_pose(image)
    if not vitpose_results:
        raise ValueError("No pose detected.")

    keypoints = vitpose_results[0]['keypoints']
    j2d_right = keypoints[-21:]
    j2d_left = keypoints[-42:-21]

    bounding_boxes = calculate_hand_bounding_boxes(j2d_right, j2d_left)

    lefthand_p_map = np.mean(j2d_left, axis=0)  # Placeholder for position map calculation
    righthand_p_map = np.mean(j2d_right, axis=0)  # Placeholder for position map calculation

    # Calculate the distance tokens and overlapping map
    distok_R2L = get_dis_tok(lefthand_p_map, righthand_p_map, bounding_boxes)
    distok_L2R = get_dis_tok(righthand_p_map, lefthand_p_map, bounding_boxes)
    O_map = get_overlapping_map(np.mean(j2d_right, axis=0), bounding_boxes[1])

    # Initialize RAT model
    rat_model = RelativeAttentionTokenization(input_dim=3, hidden_size=64, output_dim=32)
    is_righthand = True  # Assuming we are processing the right hand

    rat_tokens = rat_model(distok_R2L, distok_L2R, O_map, is_righthand)

    # Output the results
    output_image_with_bboxes(image, bounding_boxes)
    save_maps(distok_R2L, O_map)

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

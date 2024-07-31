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

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

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
    # # Set up detectron2 for hand detection
    # cfg = get_cfg()
    # cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    # cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849600/model_final_280758.pkl"
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(image)
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    outputs = detector(image)

    # Filter predictions to only keep hands (you might need a custom model or label mapping)
    pred_boxes = outputs["instances"].pred_boxes
    pred_classes = outputs["instances"].pred_classes
    hand_indices = [i for i, c in enumerate(pred_classes) if c == 1]  # Assuming '1' is the class for hands
    hand_boxes = pred_boxes[hand_indices]

    if len(hand_boxes) == 0:
        raise ValueError("No hands detected.")

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
    image_path = "/home/sjtu/eccv_workspace/hold/generator/hand_detector.d2/viz/input.jpg"  # Replace with your image path
    process_image(image_path)

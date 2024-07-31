
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

def get_position_map(bbox, H=256, W=256):
    cx, cy, sx, sy = bbox
    C = np.zeros((H, W, 2))
    for i in range(W):
        for j in range(H):
            C[j, i] = [cx + (2*i - W) * sx / (2*W), cy + (2*j - H) * sy / (2*H)]
    return C

def get_relative_distance_map(pos_map_left, pos_map_right, bbox_scale, tau=0.5):
    rel_distance = tau * (pos_map_right - pos_map_left) / bbox_scale
    return torch.sigmoid(torch.tensor(rel_distance))

def inside(patch, Bbox):
    edge_x = Bbox[0] - Bbox[2] / 2, Bbox[0] + Bbox[2] / 2
    edge_y = Bbox[1] - Bbox[3] / 2, Bbox[1] + Bbox[3] / 2
    return int(edge_x[0] < patch[0] < edge_x[1] and edge_y[0] < patch[1] < edge_y[1])

def get_overlapping_map(pos_map, bbox):
    H, W, _ = pos_map.shape
    O_map = np.zeros((H, W))
    for i in range(W):
        for j in range(H):
            O_map[j, i] = inside(pos_map[j, i], bbox)
    return O_map


def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    #detector = load_hamer(DEFAULT_CHECKPOINT)
    from detectron2.config import LazyConfig
    import hamer
    cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    img_paths = [image_path]
    for img_path in tqdm(img_paths):
        img_cv2 = cv2.imread(str(img_path))

        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img_cv2,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Calculating the relative distance map and overlapping map
        lefthand_bbox = boxes[0]
        righthand_bbox = boxes[1]

        lefthand_p_map = get_position_map(lefthand_bbox)
        righthand_p_map = get_position_map(righthand_bbox)

        relative_distance_map = get_relative_distance_map(lefthand_p_map, righthand_p_map, [lefthand_bbox[2], lefthand_bbox[3]])
        overlapping_map = get_overlapping_map(righthand_p_map, lefthand_bbox)

        output_image_with_bboxes(image, boxes)
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
    image_path = "/home/sjtu/eccv_workspace/hold/generator/hand_detector.d2/viz/input.jpg"  # Replace with your image path
    process_image(image_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils_detectron2 import DefaultPredictor_Lazy
from pathlib import Path
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from tqdm import tqdm
from .vitpose_model import ViTPoseModel
# import ipdb
import os



def rat(cfg):
    return RelativeAttentionTokenization(
        input_dim=cfg.MODEL.RAT.INPUT_DIM,
        hidden_size=cfg.MODEL.RAT.HIDDEN_SIZE,
        output_dim=cfg.MODEL.RAT.OUTPUT_DIM,
        t = cfg.MODEL.RAT.TAU,
    )


# def calculate_hand_bbox(j2d_r, j2d_l):
#     def get_bbox_from_keypoints(keypoints):
#         min_pt = np.min(keypoints, axis=0)
#         max_pt = np.max(keypoints, axis=0)
#         cx = (min_pt[0] + max_pt[0]) / 2
#         cy = (min_pt[1] + max_pt[1]) / 2
#         sx = max_pt[0] - min_pt[0]
#         sy = max_pt[1] - min_pt[1]
#         return np.array([cx, cy, sx, sy])
    
#     bbox_r = get_bbox_from_keypoints(j2d_r)
#     bbox_l = get_bbox_from_keypoints(j2d_l)
#     return np.array([bbox_r, bbox_l])



def get_postion_map(bbox, H = 16, W = 12) -> float:
    '''
     using the images with bounding boxes to calculate the position map of the two hands
    '''
    x_min, y_min, x_max, y_max = bbox
    cx = (x_max + x_min) / 2
    cy = (y_max + y_min) / 2
    sx = (x_max - x_min)
    sy = (y_max - y_min)
    C = np.zeros((H, W, 2))
    for i in range(W):
        for j in range(H):
            C[j, i] = [cx + (2 * i - W) * sx / (2 * W), cy + (2 * j - H) * sy / (2 * H)]
        
    return C



def get_dis_tok(lefthand_p_map: float, righthand_p_map: float, scale, t):
    '''
    Using the position map to calculate the relative distance token by activation function, the hyperparameter should be gain from the parser
    '''
    rel_distance = t * (righthand_p_map - lefthand_p_map) / scale
    dis_tok = torch.sigmoid(torch.tensor(rel_distance))
    return dis_tok

def inside(patch, Bbox):
    edge_x = Bbox[0], Bbox[2]
    edge_y = Bbox[1], Bbox[3]
    if (patch[0] < edge_x[1]) & (patch[0] > edge_x[0]) & (patch[1] < edge_y[1]) & (patch[1] > edge_y[0]):
        return 1
    else:
        return -1



def get_overlapping_map(pos_map, Bbox):
    '''
    using the image with bounding box to gain if some part of the hand is overlapped by others
    '''
    H, W, _ = pos_map.shape
    O_map = torch.zeros((H, W, 1))
    for i in range(W):
        for j in range(H):
            O_map[j, i] = inside(pos_map[j, i], Bbox)
    return O_map



# def process_img(image_tensor):
    """
    Process a single image tensor to detect hand bounding boxes and keypoints.

    Args:
        image_tensor (torch.Tensor): Input image tensor with shape [H, W, C].

    Returns:
        boxes (np.array): Bounding boxes for detected hands.
        right (np.array): Array indicating which boxes correspond to the right hand.
    """
    # Convert the tensor to a numpy array if necessary
    # if isinstance(image_tensor, torch.Tensor):
    #     image = image_tensor.permute(1, 2, 0).cpu().numpy()  # Convert from [C, H, W] to [H, W, C]
    # else:
    #     image = image_tensor
    image = image_tensor
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained = True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Detecting objects and keypoints
    det_img = detector(image)
    img = image.copy()[:, :, ::-1]  # Convert to RGB format

    det_instances = det_img['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

    vitposes_out = cpm.predict_pose(
        img,
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
        return None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    return boxes, right

    # img_paths = []
    # all_files = os.listdir(img_path)
    # for files in all_files:
    #     img_paths.append(os.path.join(img_path ,files))


    # for img_path in tqdm(img_paths):
    #     img_cv2 = cv2.imread(str(img_path))

    #     det_img = detector(img_cv2)
    #     img = img_cv2.copy()[:, :, ::-1]

    #     det_instances = det_img['instances']
    #     valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    #     pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    #     pred_scores = det_instances.scores[valid_idx].cpu().numpy()


    #     vitposes_out = cpm.predict_pose(
    #         img_cv2,
    #         [np.concatenate([pred_bboxes, pred_scores[:, None]], axis = 1)],
    #     )


    #     bboxes = []
    #     is_right = []

    #     # Use hands based on hand keypoint detections
    #     for vitposes in vitposes_out:
    #         left_hand_keyp = vitposes['keypoints'][-42:-21]
    #         right_hand_keyp = vitposes['keypoints'][-21:]

    #         # Rejecting not confident detections
    #         keyp = left_hand_keyp
    #         valid = keyp[:, 2] > 0.5
    #         if sum(valid) > 3:
    #             bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
    #             bboxes.append(bbox)
    #             is_right.append(0)
    #         keyp = right_hand_keyp
    #         valid = keyp[:, 2] > 0.5
    #         if sum(valid) > 3:
    #             bbox = [keyp[valid, 0].min(), keyp[valid, 1].min(), keyp[valid, 0].max(), keyp[valid, 1].max()]
    #             bboxes.append(bbox)
    #             is_right.append(1)

    #     if len(bboxes) == 0:
    #         continue

    #     boxes = np.stack(bboxes)
    #     right = np.stack(is_right)

    #     # Calculating the relative distance map and overlapping map
    #     # lefthand_bbox = boxes[0]
    #     # righthand_bbox = boxes[1]

    #     return boxes, right
def convert_tensor_to_numpy(tensor):
    # Ensure the tensor is detached and on the CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # Convert to numpy array
    numpy_array = tensor.numpy()
    return numpy_array

def process_img(img_cv2):
    detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
    detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
    detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Process the single image
    det_img = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_img['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    pred_scores = det_instances.scores[valid_idx].cpu().numpy()

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
        return None, None

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Returning the bounding boxes and the indication of right hand
    return boxes, right


def concat_maps(dis_tok_r2l, dis_tok_l2r, O_map, is_right_hand: bool):
    if is_right_hand:
        arrays = []
        arrays.append(dis_tok_r2l) 
        arrays.append(dis_tok_l2r)
        arrays.append(O_map)
        concated_map = np.concatenate(arrays, axis=-1)
        concated_map = torch.from_numpy(concated_map).float()
        return concated_map
    else:
        arrays = []
        arrays.append(dis_tok_l2r)
        arrays.append(dis_tok_r2l)
        arrays.append(O_map)
        concated_map = np.concatenate(arrays, axis=-1)
        concated_map = torch.from_numpy(concated_map).float()
        return concated_map
        
            

class Mlp(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int):
        super().__init__()
        if input_dim < 0:
            raise ValueError(f"input_dim should be positive, but got {input_dim}")
        if output_dim < 0:
            raise ValueError(f"output_dim should be positive, but got {output_dim}")
        
        '''
        define the network, can also be configured in the parser
        Sequential{
        layer1: Linear(,) relu
        layer2: Linear(,) relu
        }
        self.sequential = config.Sequential    upload model structure
        '''
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_dim)
        
    def forward(self, x):

        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))

        x = F.relu(self.fc3(x))

        return x
    

class RelativeAttentionTokenization(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int, t):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_zie = hidden_size
        self.output_dim = output_dim
        self.t = t

    def forward(self, image_path):
        # TODO the process of a single image have worked, but the batch process remains to be explored
        boxes, _ = process_img(image_path)
        lh_box = boxes[0]
        rh_box = boxes[1]

        p_map_l, p_map_r = get_postion_map(lh_box), get_postion_map(rh_box)

        tau = self.t
        rh_scale = [rh_box[2], rh_box[3]]
        lh_scale = [lh_box[2], lh_box[3]]

        r2l_dis_tok, l2r_dis_tok = get_dis_tok(p_map_r, p_map_l, rh_scale, tau), get_dis_tok(p_map_l, p_map_r, lh_scale, tau)

        r2l_O_map, l2r_O_map = get_overlapping_map(p_map_r, lh_box), get_overlapping_map(p_map_l, rh_box)


        all_map_r2l = concat_maps(r2l_dis_tok, l2r_dis_tok, r2l_O_map, is_right_hand=1)
        all_map_l2r = concat_maps(r2l_dis_tok, l2r_dis_tok, l2r_O_map, is_right_hand=0)

        mlp_in = self.input_dim
        mlp_hidden = self.hidden_zie
        mlp_out = self.output_dim

        tokenization_mlp = Mlp(mlp_in, mlp_hidden, mlp_out)

        r2l_token = tokenization_mlp(all_map_r2l)
        l2r_token = tokenization_mlp(all_map_l2r)

        return r2l_token, l2r_token



        




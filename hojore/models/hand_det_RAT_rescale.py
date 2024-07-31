import numpy as np
import cv2
import torch
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from vitpose_model import ViTPoseModel
from hamer.utils import recursive_to

def process_image(image_path, model_cfg, rescale_factor=2.5, output_folder="output"):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    detector = load_hamer(DEFAULT_CHECKPOINT)
    cpm = ViTPoseModel(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    img_cv2 = cv2.imread(image_path)

    # Detect humans in image
    det_out = detector(img_cv2)
    img = img_cv2.copy()[:, :, ::-1]

    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
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
        return

    boxes = np.stack(bboxes)
    right = np.stack(is_right)

    # Normalize bounding boxes and prepare data
    dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

    for batch in dataloader:
        batch = recursive_to(batch, device)
        img_patch = batch['img']
        for idx, patch in enumerate(img_patch):
            patch_np = patch.permute(1, 2, 0).cpu().numpy()
            patch_rescaled = (patch_np * DEFAULT_STD + DEFAULT_MEAN) * 255
            patch_rescaled = patch_rescaled.clip(0, 255).astype(np.uint8)

            # Save the rescaled bounding box image
            output_path = f"{output_folder}/rescaled_bbox_{idx}.png"
            cv2.imwrite(output_path, patch_rescaled[:, :, ::-1])
            print(f"Saved rescaled bounding box image at {output_path}")

# Define necessary model configuration variables
model_cfg = {}  # Placeholder for the actual model configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Example usage
process_image("/path/to/image.jpg", model_cfg)

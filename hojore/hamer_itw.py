import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import build_mano_head
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss, InterhandJLoss, InterhandVLoss, MaxMSELoss
from . import MANO

from .components import rat
from .components.sir import SIR, initialize_sir_parameters
from .components.cropped_image import *

log = get_pylogger(__name__)

class HAMER(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup HAMER model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=['init_renderer'])

        self.cfg = cfg

        # 在这里定义rat的参数设置
        self.rat = rat(cfg)
        # 初始化sir
        #self.sir = sir(cfg)
        self.sir = SIR(
            input_dim=cfg.MODEL.SIR.INPUT_DIM,
            hidden_dim=cfg.MODEL.SIR.HIDDEN_SIZE,
            num_heads=cfg.MODEL.SIR.NUM_HEADS,
            num_layers=cfg.MODEL.SIR.NUM_LAYERS,
            output_dim=cfg.MODEL.SIR.OUTPUT_DIM,
        )

        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            log.info(f'Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
            # self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'])
            # Load the pre-trained weights with strict=False
            self.backbone.load_state_dict(torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict'], strict=False)
            self.sir.apply(initialize_sir_parameters)


            # # Load the Pre-trained Weights for the Unmodified Parts
            # pretrained_dict = torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location='cpu')['state_dict']
            # # Get the current model's state_dict
            # model_dict = self.backbone.state_dict()
            # # Filter out keys for the new modules that do not exist in the pre-trained weights
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            #  # Update the current model's state_dict with the pre-trained weights
            # model_dict.update(pretrained_dict)
            # self.backbone.load_state_dict(model_dict)

        # Create MANO head
        self.mano_head = build_mano_head(cfg)

        # Create discriminator GAN-based transformer
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type='l1')
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type='l1')
        self.mano_parameter_loss = ParameterLoss()
        # TODO HOI Loss / Interhand Loss / Spatio-temporal Module
        # self.hoi_loss = HOILoss()
        # self.inter_j_loss = InterhandJLoss(loss_type='l2')
        # self.inter_v_loss = InterhandVLoss(loss_type='l2')
        # self.max_mse_loss = MaxMSELoss(loss_type='l2')

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k,v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer('initialized', torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def get_parameters(self):
        # TODO The meaning of all_params is not clear, should be modified after, The Params should be all put into the optimizer
        all_params = list(self.mano_head.parameters())
        all_params += list(self.backbone.parameters())
        # all_params += list(self.rat.parameters())
        return all_params

    def configure_optimizers(self) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """
        Setup model and distriminator Optimizers
        Returns:
            Tuple[torch.optim.Optimizer, torch.optim.Optimizer]: Model and discriminator optimizers
        """
        param_groups = [{'params': filter(lambda p: p.requires_grad, self.get_parameters()), 'lr': self.cfg.TRAIN.LR}]

        optimizer = torch.optim.AdamW(params=param_groups,
                                        # lr=self.cfg.TRAIN.LR,
                                        weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        optimizer_disc = torch.optim.AdamW(params=self.discriminator.parameters(),
                                            lr=self.cfg.TRAIN.LR,
                                            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

        return optimizer, optimizer_disc

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """

        # Use RGB image as input
        images = batch['jpg']
        print(f'images type: {images.type}')
        print(f'images shape: {images.shape}')
        batch_size = images.shape[0]
        # batch_size = batch['img_og'].shape[0]
        # original_image = batch['img_og']
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        # x = batch['img']
        # print(f'x type{x.type}')
        # print(f'x shape{x.shape}')
        # print(f'vit input shape{x[:,:,:,32:-32].shape}')
        # conditioning_feats = self.backbone(x[:,:,:,32:-32])
        # print(f'conditioning feats type{conditioning_feats.type}')
        # print(f'conditioning feats shape{conditioning_feats.shape}')
        # breakpoint()
        # pred_mano_params, pred_cam, _ = self.mano_head(conditioning_feats)
        lh_cropped_img = []
        rh_cropped_img = []
        rh_rat_feats = []
        lh_rat_feats = []
        # Determine the device: Use GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for i in range(batch_size):
            # Load the original image from the file using the file name
            # image_file = batch['imgname'][i]
            # image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            image_tensor = images[i]
            # Convert to NumPy array
            image_np = image_tensor.cpu().numpy()
            # image_np = np.transpose(image_np, (1, 2, 0))  # Convert from (C, H, W) to (H, W, C)
            print(image_np.dtype)
            print(f'image_np shape: {image_np.shape}') # H * W * C
            # image_np = image.cpu().numpy()

            # if isinstance(image, torch.Tensor):
            # # Move the tensor to CPU if it's on the GPU and convert to NumPy
            #     image = image.cpu().numpy()
            #     # Ensure the image is in the correct format (e.g., HWC format and dtype=uint8)
            #     image = image.transpose(1, 2, 0)  # Convert from CHW (PyTorch format) to HWC (OpenCV format)
            #     image = image.astype(np.uint8)  # Ensure the dtype is uint8


            cv2.imwrite("/home/sjtu/eccv_workspace/hamer/hamer/models/components/test_crop/output_image.jpg", image_np)
            if image_np is None:
                # raise IOError(f"Failed to read {image_file}")
                print("error reading image")
            # convert tensor to numpy for detection
            # image_numpy = image.cpu().numpy()
            # image_numpy = image_numpy[:, :, ::-1]

            lh_cropped_img_tmp, rh_cropped_img_tmp = crop_image(image_np)

            output_dir = "/home/sjtu/eccv_workspace/hamer/hamer/models/components/test_crop"  # Specify your output directory here

            # # Save the left-hand image
            # lh_image_path = os.path.join(output_dir, "left_hand.png")
            # cv2.imwrite(lh_image_path, lh_cropped_img_tmp)
            # print(f"Left hand image saved to {lh_image_path}")
            # print(f"Left hand image shape{lh_cropped_img_tmp.shape}")
            # # Save the right-hand image
            # rh_image_path = os.path.join(output_dir, "right_hand.png")
            # cv2.imwrite(rh_image_path, rh_cropped_img_tmp)
            # print(f"Right hand image saved to {rh_image_path}")
            # print(f"Right hand image shape{rh_cropped_img_tmp.shape}")

            # # lh_cropped_img, rh_cropped_img = np.stack(lh_cropped_img, lh_cropped_img_tmp), np.stack(rh_cropped_img, rh_cropped_img_tmp)
            # lh_cropped_img.append(lh_cropped_img_tmp)
            # rh_cropped_img.append(rh_cropped_img_tmp)
            # Check if the left-hand image is not empty
            target_size = (256, 192)
            blank_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)  # Create a blank image
            if lh_cropped_img_tmp is not None and lh_cropped_img_tmp.size > 0:
                # Save the left-hand image
                lh_image_path = os.path.join(output_dir, "left_hand.jpg")
                cv2.imwrite(lh_image_path, lh_cropped_img_tmp)
                print(f"Left hand image saved to {lh_image_path}")
                print(f"Left hand image shape: {lh_cropped_img_tmp.shape}")
                lh_cropped_img.append(lh_cropped_img_tmp)
            else:
                print("Left hand image is empty or not detected.Adding a blank image.")
                lh_cropped_img.append(blank_image)

            # Check if the right-hand image is not empty
            if rh_cropped_img_tmp is not None and rh_cropped_img_tmp.size > 0:
                # Save the right-hand image
                rh_image_path = os.path.join(output_dir, "right_hand.jpg")
                cv2.imwrite(rh_image_path, rh_cropped_img_tmp)
                print(f"Right hand image saved to {rh_image_path}")
                print(f"Right hand image shape: {rh_cropped_img_tmp.shape}")
                rh_cropped_img.append(rh_cropped_img_tmp)
            else:
                print("Right hand image is empty or not detected.Adding a blank image.")
                rh_cropped_img.append(blank_image)

            # Optionally handle the case where both are empty; no need
            # if (lh_cropped_img_tmp is None or lh_cropped_img_tmp.size == 0) and \
            # (rh_cropped_img_tmp is None or rh_cropped_img_tmp.size == 0):
            #     print("Both left-hand and right-hand images are empty or not detected.")
            #     lh_cropped_img.append(blank_image)
            #     rh_cropped_img.append(blank_image)
            # rat_x = np.squeeze(image_np, axis=0)
            rat_x = image_np.copy()
            print(f"rat_x shape{rat_x.shape}")
            rh_rat_feats_tmp, lh_rat_feats_tmp = self.rat(rat_x[:, :, :])
            # rh_rat_feats, lh_rat_feats = torch.stack(rh_rat_feats_tmp, rh_rat_feats), torch.stack(lh_rat_feats, lh_rat_feats_tmp)
            # Append the features to the lists
            rh_rat_feats.append(rh_rat_feats_tmp.to(device))
            lh_rat_feats.append(lh_rat_feats_tmp.to(device))
        # Free up memory from the original image tensor
        del images, image_tensor, lh_cropped_img_tmp, rh_cropped_img_tmp, rat_x
        torch.cuda.empty_cache()

        # # stack the images
        # lh_cropped_img = np.stack(lh_cropped_img)
        # rh_cropped_img = np.stack(rh_cropped_img)
        target_size = (256, 192)
        if lh_cropped_img:
            lh_cropped_img = np.stack(lh_cropped_img)
        else:
            # Handle the case where no left-hand images are found
            lh_cropped_img = np.empty((0, *target_size, 3), dtype=np.uint8)

        if rh_cropped_img:
            rh_cropped_img = np.stack(rh_cropped_img)
        else:
            # Handle the case where no right-hand images are found
            rh_cropped_img = np.empty((0, *target_size, 3), dtype=np.uint8)
        # # Once all the features are collected, stack them into tensors
        # rh_rat_feats = torch.stack(rh_rat_feats)
        # lh_rat_feats = torch.stack(lh_rat_feats)
        expected_feature_shape = (batch_size, 192, 1280)
        if rh_rat_feats:
            rh_rat_feats = torch.stack(rh_rat_feats)
        else:
            # Handle the case where no right-hand features are found
            rh_rat_feats = torch.empty((0, *expected_feature_shape), dtype=torch.float32)

        if lh_rat_feats:
            lh_rat_feats = torch.stack(lh_rat_feats)
        else:
            # Handle the case where no left-hand features are found
            lh_rat_feats = torch.empty((0, *expected_feature_shape), dtype=torch.float32)
        print(f'rh rat feats shape:{rh_rat_feats.shape}')
        print(f'lh rat feats shape:{lh_rat_feats.shape}')
        # use circulation method to gain the batch RAT
        # convert np.array to tensor(same as img_patch) and then pass through vit backbone
        # Convert to PyTorch tensors
        print(f'lh_cropped_img shape{lh_cropped_img.shape}')
        print(f'rh_cropped_img shape{rh_cropped_img.shape}')
        lh_cropped_img_tensor = torch.tensor(lh_cropped_img).permute(0, 3, 2, 1).float()  # Change from (batch, height, width, channels) to (batch, channels, height, width)
        rh_cropped_img_tensor = torch.tensor(rh_cropped_img).permute(0, 3, 2, 1).float()  # Same conversion
        print(f'lh_cropped_img_tensor shape{lh_cropped_img_tensor.shape}')
        print(f'rh_cropped_img_tensor shape{rh_cropped_img_tensor.shape}')

        # Normalize the tensors if required
        # Assuming mean and std normalization values are already defined
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        lh_cropped_img_tensor = (lh_cropped_img_tensor / 255.0 - mean) / std
        rh_cropped_img_tensor = (rh_cropped_img_tensor / 255.0 - mean) / std
        
        # Free up memory after preprocessing
        del lh_cropped_img, rh_cropped_img, mean, std
        torch.cuda.empty_cache()

        # Convert to half if using mixed precision
        lh_cropped_img_tensor = lh_cropped_img_tensor.half()
        rh_cropped_img_tensor = rh_cropped_img_tensor.half()

        # Move model and tensor to the device
        self.backbone = self.backbone.to(device)
        lh_cropped_img_tensor = lh_cropped_img_tensor.to(device)
        rh_cropped_img_tensor = rh_cropped_img_tensor.to(device)
        # Pass the tensors to the backbone
        lh_cond_feats = self.backbone(lh_cropped_img_tensor)
        lh_cond_feats = torch.tensor(lh_cond_feats).permute(0, 2, 3, 1).float()
        lh_cond_feats = lh_cond_feats.to(device)
        print(f'lh_cond_feats shape:{lh_cond_feats.shape}')
        rh_cond_feats = self.backbone(rh_cropped_img_tensor)
        rh_cond_feats = torch.tensor(rh_cond_feats).permute(0, 2, 3, 1).float()
        rh_cond_feats = rh_cond_feats.to(device)
        print(f'rh_cond_feats shape:{rh_cond_feats.shape}')
        # Free up memory after using the backbone
        del lh_cropped_img_tensor, rh_cropped_img_tensor
        torch.cuda.empty_cache()

        # Concatenate the tensors along the last dimension
        rh_rat_feats = rh_rat_feats.to(device)
        rh_all_feats = torch.concat((rh_cond_feats, rh_rat_feats), dim=-1)
        rh_all_feats = rh_all_feats.to(device)
        print(f'shape of rh all feats:{rh_all_feats.shape}')
        lh_rat_feats = lh_rat_feats.to(device)
        lh_all_feats = torch.concat((lh_cond_feats, lh_rat_feats), dim=-1)
        lh_all_feats = lh_all_feats.to(device)
        print(f'shape of lh all feats:{lh_all_feats.shape}')
        # Flatten the concatenated features along the specified dimensions
        # rh_flatten_feats = torch.flatten(rh_all_feats, start_dim=0, end_dim=1)
        # rh_flatten_feats = rh_flatten_feats.to(device)
        # print(f'shape of rh flatten feats:{rh_flatten_feats.shape}')
        # lh_flatten_feats = torch.flatten(lh_all_feats, start_dim=0, end_dim=1)
        # lh_flatten_feats = lh_flatten_feats.to(device)
        # print(f'shape of lh flatten feats:{lh_flatten_feats.shape}')
        # Concatenate the flattened features along the batch dimension (axis 0)
        # ult_feats = torch.concat((rh_flatten_feats, lh_flatten_feats), dim=0)
        B, H, W, C = rh_all_feats.shape
        rh_all_feats = rh_all_feats.view(B, H * W, C)
        lh_all_feats = lh_all_feats.view(B, H * W, C)
        ult_feats = torch.concat((rh_all_feats, lh_all_feats), dim=2)
        print(f'shape of ult feats:{ult_feats.shape}')
            # Free up memory after concatenation
        del lh_cond_feats, rh_cond_feats, rh_all_feats, lh_all_feats, rh_rat_feats, lh_rat_feats
        torch.cuda.empty_cache()
        # breakpoint()
        """
        The Ultimate features keeps the dimension of 768*2560
        """

        # sir_token = self.sir(ult_feats)
        # print(f'sir token shape:{sir_token.shape}')
        # pred_mano_params, pred_cam, _ = self.mano_head(sir_token)
        pred_mano_params, pred_cam, _ = self.mano_head(ult_feats)

        # Store useful regression outputs to the output dict
        output = {}
        output['pred_cam'] = pred_cam
        output['pred_mano_params'] = {k: v.clone() for k,v in pred_mano_params.items()}

        # Compute camera translation
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=dtype)
        pred_cam_t = torch.stack([pred_cam[:, 1],
                                  pred_cam[:, 2],
                                  2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)],dim=-1)
        output['pred_cam_t'] = pred_cam_t
        output['focal_length'] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
        pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
        mano_output = self.mano(**{k: v.float() for k,v in pred_mano_params.items()}, pose2rot=False)
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices
        output['pred_keypoints_3d'] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output['pred_vertices'] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)
        pred_keypoints_2d = perspective_projection(pred_keypoints_3d,
                                                   translation=pred_cam_t,
                                                   focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE)

        output['pred_keypoints_2d'] = pred_keypoints_2d.reshape(batch_size, -1, 2)
        # TODO compute InterHand Params pred_inter_hand()
        return output

    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> torch.Tensor:
        """
        Compute losses given the input batch and the regression output
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            torch.Tensor : Total loss for current batch
        """

        pred_mano_params = output['pred_mano_params']
        pred_keypoints_2d = output['pred_keypoints_2d']
        pred_keypoints_3d = output['pred_keypoints_3d']


        batch_size = pred_mano_params['hand_pose'].shape[0]
        device = pred_mano_params['hand_pose'].device
        dtype = pred_mano_params['hand_pose'].dtype

        # Get annotations
        gt_keypoints_2d = batch['keypoints_2d']
        gt_keypoints_3d = batch['keypoints_3d']
        gt_mano_params = batch['mano_params']
        has_mano_params = batch['has_mano_params']
        is_axis_angle = batch['mano_params_is_axis_angle']

        # Compute 3D keypoint loss
        loss_keypoints_2d = self.keypoint_2d_loss(pred_keypoints_2d, gt_keypoints_2d)
        loss_keypoints_3d = self.keypoint_3d_loss(pred_keypoints_3d, gt_keypoints_3d, pelvis_id=0)

        # Compute loss on MANO parameters
        loss_mano_params = {}
        for k, pred in pred_mano_params.items():
            gt = gt_mano_params[k].view(batch_size, -1)
            if is_axis_angle[k].all():
                gt = aa_to_rotmat(gt.reshape(-1, 3)).view(batch_size, -1, 3, 3)
            has_gt = has_mano_params[k]
            loss_mano_params[k] = self.mano_parameter_loss(pred.reshape(batch_size, -1), gt.reshape(batch_size, -1), has_gt)

        loss = self.cfg.LOSS_WEIGHTS['KEYPOINTS_3D'] * loss_keypoints_3d+\
               self.cfg.LOSS_WEIGHTS['KEYPOINTS_2D'] * loss_keypoints_2d+\
               sum([loss_mano_params[k] * self.cfg.LOSS_WEIGHTS[k.upper()] for k in loss_mano_params])

        losses = dict(loss=loss.detach(),
                      loss_keypoints_2d=loss_keypoints_2d.detach(),
                      loss_keypoints_3d=loss_keypoints_3d.detach())

        for k, v in loss_mano_params.items():
            losses['loss_' + k] = v.detach()

        output['losses'] = losses

        return loss

    # Tensoroboard logging should run from first rank only
    @pl.utilities.rank_zero.rank_zero_only
    def tensorboard_logging(self, batch: Dict, output: Dict, step_count: int, train: bool = True, write_to_summary_writer: bool = True) -> None:
        """
        Log results to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            output (Dict): Dictionary containing the regression output
            step_count (int): Global training step count
            train (bool): Flag indicating whether it is training or validation mode
        """

        mode = 'train' if train else 'val'
        batch_size = batch['keypoints_2d'].shape[0]
        images = batch['jpg']
        images = images * torch.tensor([0.229, 0.224, 0.225], device=images.device).reshape(1,3,1,1)
        images = images + torch.tensor([0.485, 0.456, 0.406], device=images.device).reshape(1,3,1,1)
        #images = 255*images.permute(0, 2, 3, 1).cpu().numpy()

        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)
        pred_vertices = output['pred_vertices'].detach().reshape(batch_size, -1, 3)
        focal_length = output['focal_length'].detach().reshape(batch_size, 2)
        gt_keypoints_3d = batch['keypoints_3d']
        gt_keypoints_2d = batch['keypoints_2d']
        losses = output['losses']
        pred_cam_t = output['pred_cam_t'].detach().reshape(batch_size, 3)
        pred_keypoints_2d = output['pred_keypoints_2d'].detach().reshape(batch_size, -1, 2)

        if write_to_summary_writer:
            summary_writer = self.logger.experiment
            for loss_name, val in losses.items():
                summary_writer.add_scalar(mode +'/' + loss_name, val.detach().item(), step_count)
        num_images = min(batch_size, self.cfg.EXTRA.NUM_LOG_IMAGES)

        gt_keypoints_3d = batch['keypoints_3d']
        pred_keypoints_3d = output['pred_keypoints_3d'].detach().reshape(batch_size, -1, 3)

        # We render the skeletons instead of the full mesh because rendering a lot of meshes will make the training slow.
        #predictions = self.renderer(pred_keypoints_3d[:num_images],
        #                            gt_keypoints_3d[:num_images],
        #                            2 * gt_keypoints_2d[:num_images],
        #                            images=images[:num_images],
        #                            camera_translation=pred_cam_t[:num_images])
        predictions = self.mesh_renderer.visualize_tensorboard(pred_vertices[:num_images].cpu().numpy(),
                                                               pred_cam_t[:num_images].cpu().numpy(),
                                                               images[:num_images].cpu().numpy(),
                                                               pred_keypoints_2d[:num_images].cpu().numpy(),
                                                               gt_keypoints_2d[:num_images].cpu().numpy(),
                                                               focal_length=focal_length[:num_images].cpu().numpy())
        if write_to_summary_writer:
            summary_writer.add_image('%s/predictions' % mode, predictions, step_count)

        return predictions

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)

    def training_step_discriminator(self, batch: Dict,
                                    hand_pose: torch.Tensor,
                                    betas: torch.Tensor,
                                    optimizer: torch.optim.Optimizer) -> torch.Tensor:
        """
        Run a discriminator training step
        Args:
            batch (Dict): Dictionary containing mocap batch data
            hand_pose (torch.Tensor): Regressed hand pose from current step
            betas (torch.Tensor): Regressed betas from current step
            optimizer (torch.optim.Optimizer): Discriminator optimizer
        Returns:
            torch.Tensor: Discriminator loss
        """
        batch_size = hand_pose.shape[0]
        gt_hand_pose = batch['hand_pose']
        gt_betas = batch['betas']
        gt_rotmat = aa_to_rotmat(gt_hand_pose.view(-1,3)).view(batch_size, -1, 3, 3)
        disc_fake_out = self.discriminator(hand_pose.detach(), betas.detach())
        loss_fake = ((disc_fake_out - 0.0) ** 2).sum() / batch_size
        disc_real_out = self.discriminator(gt_rotmat, gt_betas)
        loss_real = ((disc_real_out - 1.0) ** 2).sum() / batch_size
        loss_disc = loss_fake + loss_real
        loss = self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_disc
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return loss_disc.detach()

    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:
        """
        Run a full training step
        Args:
            joint_batch (Dict): Dictionary containing image and mocap batch data
            batch_idx (int): Unused.
            batch_idx (torch.Tensor): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        batch = joint_batch['img']
        mocap_batch = joint_batch['mocap']
        optimizer = self.optimizers(use_pl_optimizer=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            optimizer, optimizer_disc = optimizer

        batch_size = batch['jpg'].shape[0]
        output = self.forward_step(batch, train=True)
        pred_mano_params = output['pred_mano_params']
        if self.cfg.get('UPDATE_GT_SPIN', False):
            self.update_batch_gt_spin(batch, output)
        loss = self.compute_loss(batch, output, train=True)
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            disc_out = self.discriminator(pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1))
            loss_adv = ((disc_out - 1.0) ** 2).sum() / batch_size
            loss = loss + self.cfg.LOSS_WEIGHTS.ADVERSARIAL * loss_adv

        # Error if Nan
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')

        optimizer.zero_grad()
        self.manual_backward(loss)
        # Clip gradient
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(), self.cfg.TRAIN.GRAD_CLIP_VAL, error_if_nonfinite=True)
            self.log('train/grad_norm', gn, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        optimizer.step()
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            loss_disc = self.training_step_discriminator(mocap_batch, pred_mano_params['hand_pose'].reshape(batch_size, -1), pred_mano_params['betas'].reshape(batch_size, -1), optimizer_disc)
            output['losses']['loss_gen'] = loss_adv
            output['losses']['loss_disc'] = loss_disc

        if self.global_step > 0 and self.global_step % self.cfg.GENERAL.LOG_STEPS == 0:
            self.tensorboard_logging(batch, output, self.global_step, train=True)

        self.log('train/loss', output['losses']['loss'], on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return output

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0) -> Dict:
        """
        Run a validation step and log to Tensorboard
        Args:
            batch (Dict): Dictionary containing batch data
            batch_idx (int): Unused.
        Returns:
            Dict: Dictionary containing regression output.
        """
        # batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, output, train=False)
        output['loss'] = loss
        self.tensorboard_logging(batch, output, self.global_step, train=False)

        return output

def forward_step(self, batch: Dict, train: bool = False) -> Dict:
    """
    Run a forward step of the network
    Args:
        batch (Dict): Dictionary containing batch data
        train (bool): Flag indicating whether it is training or validation mode
    Returns:
        Dict: Dictionary containing the regression output
    """

    batch_size = batch['jpg'].shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 获取左手和右手裁剪图像
    lh_cropped_img = batch['lh_cropped_img']
    rh_cropped_img = batch['rh_cropped_img']

    # 如果左手裁剪图像为空，则跳过对其的处理
    if lh_cropped_img.shape[0] > 0:
        lh_cropped_img_tensor = torch.tensor(lh_cropped_img).permute(0, 3, 2, 1).float().to(device)
        lh_cropped_img_tensor = (lh_cropped_img_tensor / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        lh_cropped_img_tensor = lh_cropped_img_tensor.half()

        # 提取左手特征
        lh_cond_feats = self.backbone(lh_cropped_img_tensor)
        lh_cond_feats = lh_cond_feats.permute(0, 2, 3, 1).float()
    else:
        lh_cond_feats = torch.empty((0, 16, 12, 1280), device=device)

    # 如果右手裁剪图像为空，则跳过对其的处理
    if rh_cropped_img.shape[0] > 0:
        rh_cropped_img_tensor = torch.tensor(rh_cropped_img).permute(0, 3, 2, 1).float().to(device)
        rh_cropped_img_tensor = (rh_cropped_img_tensor / 255.0 - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        rh_cropped_img_tensor = rh_cropped_img_tensor.half()

        # 提取右手特征
        rh_cond_feats = self.backbone(rh_cropped_img_tensor)
        rh_cond_feats = rh_cond_feats.permute(0, 2, 3, 1).float()
    else:
        rh_cond_feats = torch.empty((0, 16, 12, 1280), device=device)

    # 获取RAT特征
    rh_rat_feats = batch['rh_rat_feats'].to(device)
    lh_rat_feats = batch['lh_rat_feats'].to(device)

    # 拼接手部特征和RAT特征
    if rh_cond_feats.shape[0] > 0:
        rh_all_feats = torch.cat((rh_cond_feats, rh_rat_feats), dim=-1)
    else:
        rh_all_feats = torch.empty((0, 16, 12, 2560), device=device)

    if lh_cond_feats.shape[0] > 0:
        lh_all_feats = torch.cat((lh_cond_feats, lh_rat_feats), dim=-1)
    else:
        lh_all_feats = torch.empty((0, 16, 12, 2560), device=device)

    # 展平特征
    B, H, W, C = rh_all_feats.shape
    rh_all_feats = rh_all_feats.view(B, H * W, C)
    lh_all_feats = lh_all_feats.view(B, H * W, C)

    # 最终拼接的特征
    ult_feats = torch.cat((rh_all_feats, lh_all_feats), dim=2)

    # 清理不再需要的tensor
    del lh_cond_feats, rh_cond_feats, rh_all_feats, lh_all_feats, rh_rat_feats, lh_rat_feats
    torch.cuda.empty_cache()

    # SIR处理和MANO头部预测
    sir_token = self.sir(ult_feats)
    pred_mano_params, pred_cam, _ = self.mano_head(sir_token)

    # Store useful regression outputs to the output dict
    output = {}
    output['pred_cam'] = pred_cam
    output['pred_mano_params'] = {k: v.clone() for k, v in pred_mano_params.items()}

    # Compute camera translation
    focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(batch_size, 2, device=device, dtype=torch.float32)
    pred_cam_t = torch.stack([pred_cam[:, 1],
                              pred_cam[:, 2],
                              2*focal_length[:, 0]/(self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] +1e-9)], dim=-1)
    output['pred_cam_t'] = pred_cam_t
    output['focal_length'] = focal_length

    # Compute model vertices, joints and the projected joints
    pred_mano_params['global_orient'] = pred_mano_params['global_orient'].reshape(batch_size, -1, 3, 3)
    pred_mano_params['hand_pose'] = pred_mano_params['hand_pose'].reshape(batch_size, -1, 3, 3)
    pred_mano_params['betas'] = pred_mano_params['betas'].reshape(batch_size, -1)
    mano_output = self.mano(**{k: v.float() for k, v in pred_mano_params.items()}, pose2rot=False)
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

    return output

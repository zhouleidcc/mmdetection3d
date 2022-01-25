import torch
import onnx
import time
import numpy as np
import mmcv
from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

from mmdet3d.core import (Box3DMode, CameraInstance3DBoxes,
                          DepthInstance3DBoxes, LiDARInstance3DBoxes,
                          show_multi_modality_result, show_result,
                          show_seg_result, box3d_multiclass_nms)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

onnx_file = './fcos3d_928_1600_modified.onnx'
trt_file = './fcos3d_928_1600_modified.trt'
config_file = '../configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py'
cfg = mmcv.Config.fromfile(config_file).model['test_cfg']
onnx_model = onnx.load(onnx_file)

## Model input
# inputs = torch.rand(1, 3, 928, 1600).cuda()
inputs = torch.from_numpy(np.load('./image.npy')).cuda()
## Model input shape info
opt_shape_dict = {
    'input': [list(inputs.shape),
              list(inputs.shape),
              list(inputs.shape)]
}

## Create TensorRT engine
if False:
    max_workspace_size = 1 << 30
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
    max_workspace_size=max_workspace_size)

    ## Save TensorRT engine
    save_trt_engine(trt_engine, trt_file)

## Run inference with TensorRT

since = time.time()
trt_model = TRTWrapper(trt_file, ['input'], ['boxes_3d', 'scores_3d', 'bev_boxes', 'dir_scores', 'attr_scores'])
time_elapsed = time.time() - since
print(time_elapsed)

with torch.no_grad():
    trt_outputs = trt_model({'input': inputs})
    boxes_3d = trt_outputs['boxes_3d']
    scores_3d = trt_outputs['scores_3d']
    bev_boxes = trt_outputs['bev_boxes']
    dir_scores = trt_outputs['dir_scores']
    attr_scores = trt_outputs['attr_scores']
    max_score, _ = scores_3d.max(dim=0)
    _, index = max_score.topk(400)
    boxes_3d = boxes_3d[:, index].permute(1, 0)
    scores_3d = scores_3d[:, index].permute(1, 0)
    bev_boxes = bev_boxes[:, index].permute(1, 0)
    dir_scores = dir_scores[index]
    attr_scores = attr_scores[index]

    results = box3d_multiclass_nms(boxes_3d, bev_boxes,
                         scores_3d, cfg.score_thr,
                         cfg.max_per_img, cfg, dir_scores,
                         attr_scores)
    pred_bboxes = results[0].cpu().numpy()
    pred_scores = results[1].cpu().numpy()
    img_filename = './data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg'
    file_name = 'n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_trt.png'
    out_dir = './demo'
    img = mmcv.imread(img_filename)
    instrinc = np.load('./intrinsic_params.npy').tolist()
    # filter out low score bboxes for visualization
    cfg.score_thr = 0.3
    if cfg.score_thr > 0:
        inds = pred_scores > cfg.score_thr
        pred_bboxes = pred_bboxes[inds]

    show_bboxes = CameraInstance3DBoxes(
        pred_bboxes, box_dim=pred_bboxes.shape[-1], origin=(0.5, 1.0, 0.5))

    show_multi_modality_result(
        img,
        None,
        show_bboxes,
        instrinc,
        out_dir,
        file_name,
        box_mode='camera',
        show=True)

    print('ok')
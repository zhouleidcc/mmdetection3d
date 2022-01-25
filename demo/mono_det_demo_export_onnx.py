# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
import torch
from functools import partial
import numpy as np
def check_model_is_cuda(model):
    return next(model.parameters()).is_cuda

def export_onnx(model, data, model_name):
    img_list, img_meta_list = [data['img']], data['img_metas']
    data_shape = np.array(img_meta_list[0][0]['batch_input_shape'])
    strides = model.bbox_head.strides
    feat_shapes = np.ceil(np.array([data_shape / strid for strid in strides])).astype(np.int32)
    points = np.array([a[0] * a[1] for a in feat_shapes])
    num_points = np.sum(points)
    model.forward = partial(
        model.forward,
        img_metas=img_meta_list,
        return_loss=False,
        rescale=False)

    input = data['img'][0]
    input_names = ["input"]
    out_names = ["boxes_3d",
                 "scores_3d",
                 "bev_boxes",
                 "dir_scores",
                 "attr_scores"]
    input_shape = [tuple(a for a in input.shape)]
    if input.dtype == torch.float32 and model.fp16_enabled:
        model.fp16_enabled = False
    example_outputs = [(input.new_ones(9, num_points),
                        input.new_ones(11, num_points),
                        input.new_ones(5, num_points),
                        input.new_ones(num_points),
                        input.new_ones(num_points),)]
    onnx_file = '%s_%d_%d.onnx' % (model_name, input_shape[0][2], input_shape[0][3])
    # data['img_metas'] = tuple(data['img_metas'])
    # data['img'] = tuple(data['img'])
    cv2onnx(model, tuple(img_list), example_outputs, onnx_file,
            input_names=input_names, output_names=out_names)

def cv2onnx(model, data, example_outputs, onnx_file,
            input_names=[ "input" ], output_names=["output"], verbose=False, enable_onnx_checker=True):
    """
    convert torch model to tflite model using onnx
    """
    # if type(input_shape[0]) == tuple:
    #     if check_model_is_cuda(model):
    #         dummy_input = tuple([torch.randn(ishape, device="cuda") for ishape in input_shape])
    #     else:
    #         dummy_input = tuple([torch.randn(ishape, device="cpu") for ishape in input_shape])
    # elif type(input_shape) == tuple:
    #     if check_model_is_cuda(model):
    #         dummy_input = torch.randn(input_shape, device="cuda")
    #     else:
    #         dummy_input = torch.randn(input_shape, device="cpu")
    # else:
    #     raise Exception("input_shape must be tuple")
    opset_version = 12
    try:
        # training = torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
        # do_constant_folding = (not train) and (not next(model.parameters()).is_cuda)
        # test = (not next(model.parameters()).is_cuda)
        torch.onnx.export(model, data, onnx_file,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=(not next(model.parameters()).is_cuda),
                          opset_version=opset_version,
                          input_names=input_names ,
                          output_names=output_names,
                          verbose=verbose,
                          example_outputs=example_outputs,
                          dynamic_axes=None,
                          enable_onnx_checker=enable_onnx_checker)
    except RuntimeError as e:
        torch.onnx.export(model, data, onnx_file,
                          opset_version=opset_version,
                          input_names=input_names , output_names=output_names,
                          example_outputs=example_outputs)

def main():
    parser = ArgumentParser()
    parser.add_argument('--image', default='./data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525.jpg', help='image file')
    parser.add_argument('--ann', default= './data/nuscenes/n015-2018-07-24-11-22-45+0800__CAM_BACK__1532402927637525_mono3d.coco.json', help='ann file')
    parser.add_argument('--config', default='../configs/fcos3d/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune.py', help='Config file')
    parser.add_argument('--checkpoint', default='../checkpoints/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.15, help='bbox score threshold')
    parser.add_argument(
        '--out-dir', type=str, default='demo', help='dir to save results')
    parser.add_argument(
        '--show',
        action='store_true',
        help='show online visualization results')
    parser.add_argument(
        '--snapshot',
        action='store_true',
        help='whether to save online visualization results')
    parser.add_argument('--infer', type=str, default=True, help='only infer not export onnx file')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)

    # test a single image
    result, data = inference_mono_3d_detector(model, args.image, args.ann)
    for res in result:
        index = ~torch.any(torch.isnan(res['img_bbox']['boxes_3d'].tensor), dim=1)
        for k, v in res['img_bbox']['boxes_3d'].__dict__.items():
            if type(v) == torch.Tensor:
                setattr(res['img_bbox']['boxes_3d'], k, v[index])
        for k, v in res['img_bbox'].items():
            if type(v) == torch.Tensor:
                if type(v) == torch.Tensor:
                    res['img_bbox'][k] = v[index]
    # export onnx file
    if False:
        model_name = 'fcos3d'
        export_onnx(model, data, model_name)
    # show the results
    show_result_meshlab(
        data,
        result,
        args.out_dir,
        args.score_thr,
        show=args.show,
        snapshot=args.snapshot,
        task='mono-det')


if __name__ == '__main__':
    main()

import torch
import onnx
import numpy as np
from onnx import helper
import sys
IS_PYTHON3 = sys.version_info > (3,)
from mmcv.tensorrt import (TRTWrapper, onnx2trt, save_trt_engine,
                                   is_tensorrt_plugin_loaded)

assert is_tensorrt_plugin_loaded(), 'Requires to complie TensorRT plugins in mmcv'

def legacy_opset_pre_ver(version):
  return onnx.defs.onnx_opset_version() < version



def load_onnx_model(onnx_file='./fcos3d_928_1600.onnx'):
    onnx_model = onnx.load(onnx_file)
    return onnx_model

def load_onnx_graph(onnx_file='./fcos3d_928_1600.onnx'):
    onnx_model = onnx.load(onnx_file)
    return onnx_model.graph

def _onnx_graph_index(graph_prop_list, prop, by_name=False):
  for i, n in enumerate(graph_prop_list):
    if by_name:
      if n.name == prop.name:
        return i
    else:
      if n == prop:
        return i
  return -1

# inputs = torch.rand(1, 3, 928, 1600).cuda()
# ## Model input shape info
# opt_shape_dict = {
#     'input': [list(inputs.shape),
#               list(inputs.shape),
#               list(inputs.shape)]
# }

def convert_onnx_trt(opt_shape_dict, onnx_file='./fcos3d_928_1600.onnx', trt_file='fcos3d_928_1600.trt'):
    onnx_model = load_onnx_model(onnx_file)
    ## Create TensorRT engine
    max_workspace_size = 1 << 30
    trt_engine = onnx2trt(
        onnx_model,
        opt_shape_dict,
        max_workspace_size=max_workspace_size)

    ## Save TensorRT engine
    save_trt_engine(trt_engine, trt_file)
def load_trt_model(trt_file='fcos3d_928_1600.trt', input_namess=['input'], output_names=["boxes_3d", "scores_3d", "dir_scores", "attr_scores"]):
    ## Run inference with TensorRT
    trt_model = TRTWrapper(trt_file, input_namess, output_names)
    return trt_model

def trt_infer(trt_file, inputs, outputs):
    with torch.no_grad():
        trt_model = load_trt_model(trt_file, inputs, outputs)
        trt_outputs = trt_model({'input': inputs})
        boxes_3d = trt_outputs['boxes_3d']
        scores_3d = trt_outputs['scores_3d']
        dir_scores = trt_outputs['dir_scores']
        attr_scores = trt_outputs['attr_scores']

def init_input_data(input_shape=[1, 3, 928, 1600]):
    inputs = torch.rand(input_shape).cuda()
    opt_shape_dict = {
        'input': [list(inputs.shape),
                  list(inputs.shape),
                  list(inputs.shape)]
    }
    return

def _onnx_node(graph, name):
  for node in graph.node:
    if node.name == name:
      return node
    for input in node.input:
      if input == name:
        return node
    for output in node.output:
      if output == name:
        return node
  return None

def _onnx_node_input_map(node_list):
  m = {}
  for n in node_list:
    for n_input in n.input:
      if n_input in m:
        m[n_input].append(n)
      else:
        m[n_input] = [n]
  return m

def modify_node(graph, name):
    node = _onnx_node(graph, name)
    node_i = _onnx_graph_index(graph.node, node, by_name=True)


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:

    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor

def _convert_onnx_attribute_proto(attr_proto):
    '''
    Convert ONNX AttributeProto into Python object
    '''
    if attr_proto.HasField('f'):
        return attr_proto.f
    elif attr_proto.HasField('i'):
        return attr_proto.i
    elif attr_proto.HasField('s'):
        return str(attr_proto.s, 'utf-8')
    elif attr_proto.HasField('t'):
        return attr_proto.t  # this is a proto!
    elif attr_proto.floats:
        return list(attr_proto.floats)
    elif attr_proto.ints:
        return list(attr_proto.ints)
    elif attr_proto.strings:
        str_list = list(attr_proto.strings)
        str_list = list(map(lambda x: str(x, 'utf-8'), str_list))
        return str_list
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))


def convert_onnx_attribute_proto(attr_proto):
  """
  Convert an ONNX AttributeProto into an appropriate Python object
  for the type.
  NB: Tensor attribute gets returned as the straight proto.
  """
  if attr_proto.HasField('f'):
    return attr_proto.f
  elif attr_proto.HasField('i'):
    return attr_proto.i
  elif attr_proto.HasField('s'):
    return str(attr_proto.s, 'utf-8') if IS_PYTHON3 else attr_proto.s
  elif attr_proto.HasField('t'):
    return attr_proto.t  # this is a proto!
  elif attr_proto.HasField('g'):
    return attr_proto.g
  elif attr_proto.floats:
    return list(attr_proto.floats)
  elif attr_proto.ints:
    return list(attr_proto.ints)
  elif attr_proto.strings:
    str_list = list(attr_proto.strings)
    if IS_PYTHON3:
      str_list = list(map(lambda x: str(x, 'utf-8'), str_list))
    return str_list
  elif attr_proto.HasField('sparse_tensor'):
    return attr_proto.sparse_tensor
  elif not legacy_opset_pre_ver(15) and attr_proto.HasField('tp'):
    return attr_proto.tp
  else:
    raise ValueError("Unsupported ONNX attribute: {}".format(attr_proto))

def convert_onnx(attr):
  return convert_onnx_attribute_proto(attr)



if __name__ == '__main__':
    model = load_onnx_model()
    # name = '1644'
    # temp_node = _onnx_node(graph, name)
    # temp_node = dict([(attr.name,
    #                     convert_onnx_attribute_proto(attr))
    #                    for attr in temp_node.attribute])
    # if hasattr(temp_node['value'], 'SerializeToString') and callable(temp_node['value'].SerializeToString):
    #     result = temp_node['value'].SerializeToString()
    #     temp = np.frombuffer(temp_node['value'].raw_data)
    #     data = result.decode()
    #     print('ok')
    # temp_node = convert_onnx_attribute_proto(temp_node['value'])
    #
    #
    # temp_node = convert_onnx(temp_node.attribute)
    # print(temp_node)

    name = 'Resize_791'
    node = _onnx_node(model.graph, name)
    print(node)
    np_scales_weights = np.ones(shape=(4)).astype(np.float32)
    np_scales_weights[2:] = 2
    scales_weights = create_initializer_tensor(name='scales_791', tensor_array=np_scales_weights, data_type=onnx.TensorProto.FLOAT)
    model.graph.initializer.append(scales_weights)
    node_new = helper.make_node(
        'Resize',  # op name
        ['1642', '', 'scales_791'],  # inputs
        ['1647'],  # outputs
        name=name,
        coordinate_transformation_mode='asymmetric',
        cubic_coeff_a=-0.75,
        mode='nearest',
        nearest_mode='floor',  # attributes
    )
    print('...............')
    node_index = _onnx_graph_index(model.graph.node, node, by_name=True)
    model.graph.node.remove(node)
    model.graph.node.insert(node_index, node_new)
    print(node_new)



    print('...............')

    name = 'Resize_797'
    node = _onnx_node(model.graph, name)
    print(node)
    scales_weights = create_initializer_tensor(name='scales_797', tensor_array=np_scales_weights,
                                               data_type=onnx.TensorProto.FLOAT)
    model.graph.initializer.append(scales_weights)
    node_new = helper.make_node(
        'Resize',  # op name
        ['1648', '', 'scales_797'],  # inputs
        ['1653'],  # outputs
        name=name,
        coordinate_transformation_mode='asymmetric',
        cubic_coeff_a=-0.75,
        mode='nearest',
        nearest_mode='floor',  # attributes

    )
    print('...............')
    node_index = _onnx_graph_index(model.graph.node, node, by_name=True)
    model.graph.node.remove(node)
    model.graph.node.insert(node_index, node_new)
    rm_names = ['1643', '1644', '1645', '1646', '1649', '1650', '1651', '1652']
    for rm_name in rm_names:
        rm_node = _onnx_node(model.graph, rm_name)
        print(rm_node)
        model.graph.node.remove(rm_node)
    # model_def = onnx.helper.make_model(graph, producer_name="zhoulei")
    # model_def.opset_import[0].version = 12
    # onnx.checker.check_model(model_def)
    onnx.save(model, './fcos3d_928_1600_modified.onnx')
    print(node_new)
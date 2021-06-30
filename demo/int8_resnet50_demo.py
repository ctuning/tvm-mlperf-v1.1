from PIL import Image
import numpy as np
import torch
import sys

import tvm
from tvm.relay.build_module import bind_params_by_name
from tvm import relay
from tvm.relay.op.contrib.dnnl import partition_for_dnnl
from tvm.relay import transform, testing
from tvm.driver.tvmc.common import convert_graph_layout
from tvm.contrib.graph_executor import GraphModule

target = "llvm -mcpu=core-avx2"


def top5(arr):
    N = arr.shape[0]
    topN_res = []
    for i in range(N):
        arr_ = np.squeeze(arr[i,:])
        n = 5
        idxs = (-arr_).argsort()[:n]
        res = [(i, arr_[i]) for i in idxs]
        res.sort(key=lambda a: a[1], reverse=True)
        topN_res.append(res)
    return topN_res


def preprocess(img_data):
    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(img_data.shape).astype('float32')
    for i in range(img_data.shape[0]):
        # for each pixel in each channel, divide the value by 255 to get value between [0, 1] and then normalize
        norm_img_data[i, :, :] = (img_data[i, :, :]/255 - mean_vec[i]) / stddev_vec[i]
    return norm_img_data


def build(mod, params=None, name=None):
    if mod is not None:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
        if name is not None:
            lib.export_library(name)
    else:
        lib = tvm.runtime.load_module(name)
    return lib


def score(lib, dev, inputs, perf=False):
    g_module = GraphModule(lib["default"](dev))

    for idx, data in enumerate(inputs):
        g_module.set_input(idx, data)
    if perf is True:
        res = g_module.module.time_evaluator("run", dev, number=300, repeat=1)()
        res = g_module.module.time_evaluator("run", dev, number=300, repeat=1)()
        print(res)
    else:
        g_module.run()

    return g_module.get_output(0).numpy()


def get_model(model_path, batch=1):
    model = torch.load(model_path)
    model.eval()

    input_shape = [batch, 3, 224, 224]
    input_data = torch.randn(input_shape)
    scripted_model = torch.jit.trace(model, input_data).eval()
    input_name = "X"
    shape_list = [(input_name, input_shape)]

    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    mod["main"] = bind_params_by_name(mod["main"], params)

    return mod


def handle_int8_resnet():
    model_path = sys.argv[1]
    img_path = sys.argv[2]
    batch = 1
    # read kitty
    img = Image.open(img_path)
    img = img.resize(size=(224, 224))
    in_data = np.asarray(img)
    in_data = in_data.transpose((2, 0, 1))
    shape = in_data.shape
    in_data = preprocess(in_data)
    in_data = np.broadcast_to(in_data.astype("float32"), shape=(batch, *shape))

    use_prebuilt = False
    if use_prebuilt:
        mod_ref = None
        mod_dnnl = None
    else:
        mod = get_model(model_path, batch)
        #  move to NHWC layout
        mod_ref = convert_graph_layout(mod, "NHWC")
        seq = tvm.transform.Sequential([
            transform.FoldConstant(),
        ])
        with tvm.transform.PassContext(opt_level=3):
            mod_ref = seq(mod_ref)
        # partitioning for DNNL
        mod_dnnl = partition_for_dnnl(mod_ref)

    ref_lib = build(mod_ref, name=f"ref_int8_resnet50_mb{batch}.dylib")
    dnnl_lib = build(mod_dnnl, name=f"dnnl_int8_resnet50_mb{batch}.dylib")

    dev = tvm.cpu()

    ref_res = score(ref_lib, inputs=[in_data], dev=dev, perf=True)
    dnnl_res = score(dnnl_lib, inputs=[in_data], dev=dev, perf=True)

    print("=== REF ===")
    print(top5(ref_res))
    print("=== DNNL ===")
    print(top5(dnnl_res))

    out_quant = 0.2882
    tvm.testing.assert_allclose(dnnl_res, ref_res, rtol=1e-3, atol=2*out_quant)  # tolerance is +-2 quant


if __name__ == "__main__":
    handle_int8_resnet()

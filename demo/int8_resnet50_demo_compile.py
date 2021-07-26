import time

from PIL import Image
import numpy as np
import torch
import sys
import os
import platform

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
        arr_ = np.squeeze(arr[i, :])
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


def get_model(model_path, batch=1):
    model = torch.jit.load(model_path)
    model.eval()

    shape_list = [("X", [batch, 3, 224, 224])]

    mod, params = relay.frontend.from_pytorch(model, shape_list)
    mod["main"] = bind_params_by_name(mod["main"], params)

    return mod, params


def compile_resnet(model_path, export_model_path):
    batch = 1

    mod, params = get_model(model_path, batch)

    mod = transform.FoldConstant()(mod)

    #  move to NHWC layout, prerequisite for DNNL partitioning
    mod = convert_graph_layout(mod, "NHWC")
    mod = transform.FoldConstant()(mod)
    # partitioning for DNNL
    mod = partition_for_dnnl(mod)
    lib = relay.build(mod, target=target)
    lib.export_library(export_model_path)


def check_accuracy(model_path, export_model_path, img_path):
    batch = 1
    # read kitty
    img = Image.open(img_path)
    img = img.resize(size=(224, 224))
    in_data = np.asarray(img)
    in_data = in_data.transpose((2, 0, 1))
    shape = in_data.shape
    in_data = preprocess(in_data)
    in_data = np.broadcast_to(in_data.astype("float32"), shape=(batch, *shape))

    #  Torch scoring
    model = torch.jit.load(model_path).eval()
    with torch.no_grad():
        torch_out = model.forward(torch.from_numpy(np.array(in_data)))
        torch_out = torch_out.numpy()

    #  TVM scoring
    lib = tvm.runtime.load_module(export_model_path)
    dev = tvm.cpu()
    g_module = GraphModule(lib["default"](dev))
    g_module.set_input(0, in_data)
    g_module.run()
    tvm_out = g_module.get_output(0).numpy()

    print("=== Torch Top5 ===")
    print(top5(torch_out))
    print("=== TVM + DNNL Top5 ===")
    print(top5(tvm_out))

    #  Quantum value is read from last qnn.dequantize operation
    out_quantum = 0.2882
    #  Tolerance is +-2 quantum because of floating point error in requantization operations
    assert np.allclose(torch_out, tvm_out, rtol=1e-3, atol=2*out_quantum)


def main():
    img_path = "__data/cat3.png"
    model_path = "__data/resnet50_INT8bit_quantized.pt"
    os.makedirs("__prebuilt", exist_ok=True)
    so_ext = "dylib" if platform.system() == "Darwin" else "so"
    export_model_path = f"__prebuilt/dnnl_int8_resnet50.{so_ext}"

    compile_resnet(model_path, export_model_path)
    check_accuracy(model_path, export_model_path, img_path)


if __name__ == "__main__":
    main()

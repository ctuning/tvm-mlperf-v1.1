# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Unit tests for JSON codegen and runtime."""
import os
import sys

import numpy as np

import tvm
import tvm.relay.op as reg
import tvm.relay.testing
from tvm import relay, runtime
from tvm.contrib import utils
from tvm.relay import transform
from tvm.relay.backend import compile_engine
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.op.contrib.register import get_pattern_table
import tvm.relay.qnn
from tvm.relay.op.contrib.dnnl import partition_for_dnnl


def set_func_attr(func, compile_name, symbol_name):
    func = func.with_attr("Primitive", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Inline", tvm.tir.IntImm("int32", 1))
    func = func.with_attr("Compiler", compile_name)
    func = func.with_attr("global_symbol", symbol_name)
    return func


def check_result(
    mod, ref_mod, map_inputs, out_shape, tol=1e-5, target="llvm", device=tvm.cpu(), params=None,
    dtype="float32", ref_result=None, atol=None
):
    if sys.platform == "win32":
        print("Skip test on Windows for now")
        return

    if atol is None:
        atol = tol

    if ref_result is None:
        # Run the reference result
        compile_engine.get().clear()
        with tvm.transform.PassContext(opt_level=3):
            json, lib, param = relay.build(ref_mod, target=target, params=params)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, device=device, dtype=dtype)
        out = rt_mod.get_output(0, out)
        ref_result = out.numpy()
        np.set_printoptions(threshold=np.inf)

    def check_vm_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            exe = relay.vm.compile(mod, target=target, params=params)
        code, lib = exe.save()
        exe = runtime.vm.Executable.load_exec(code, lib)
        vm = runtime.vm.VirtualMachine(exe, device)
        out = vm.run(**map_inputs)
        tvm.testing.assert_allclose(out.numpy(), ref_result, rtol=tol, atol=atol)

    def check_graph_executor_result():
        compile_engine.get().clear()
        with relay.build_config(opt_level=3):
            json, lib, param = relay.build(mod, target=target, params=params)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, device)

        for name, data in map_inputs.items():
            rt_mod.set_input(name, data)
        rt_mod.set_input(**param)
        rt_mod.run()
        out = tvm.nd.empty(out_shape, dtype=dtype, device=device)
        out = rt_mod.get_output(0, out)
        tvm.testing.assert_allclose(out.numpy(), ref_result, rtol=tol, atol=atol)

    check_vm_result()
    check_graph_executor_result()


def test_conv2d():
    """Test a subgraph with a single conv2d operator."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    def conv2d_direct():
        dtype = "float32"
        ishape = (1, 32, 14, 14)
        w1shape = (32, 32, 3, 3)

        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=w1shape, dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1))

        func = relay.Function([data0, weight0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(w1shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        data0 = relay.var("data", shape=ishape, dtype=dtype)
        weight0 = relay.var("weight", shape=w1shape, dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1))
        main_f = relay.Function([data0, weight0], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w1_data}, (1, 32, 14, 14)

    def group_conv2d():
        dtype = "float32"
        ishape = (1, 32, 14, 14)
        w2shape = (32, 1, 3, 3)

        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        weight0 = relay.var("weight", shape=(w2shape), dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=32)

        func = relay.Function([data0, weight0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data = relay.var("data", shape=(ishape), dtype=dtype)
        weight = relay.var("weight", shape=(w2shape), dtype=dtype)
        main_f = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        data0 = relay.var("data", shape=(ishape), dtype=dtype)
        weight0 = relay.var("weight", shape=(w2shape), dtype=dtype)
        out = relay.nn.conv2d(data0, weight0, kernel_size=(3, 3), padding=(1, 1), groups=32)
        main_f = relay.Function([data0, weight0], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w_data = np.random.uniform(0, 1, w2shape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w_data}, (1, 32, 14, 14)

    for mod, ref_mod, map_inputs, out_shape in [conv2d_direct(), group_conv2d()]:
        check_result(mod, ref_mod, map_inputs, out_shape, tol=1e-5)


def test_add():
    """Test a subgraph with a single add operator."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    shape = (10, 10)

    def gen_add():
        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        out = relay.add(data0, data1)

        func = relay.Function([data0, data1], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        main_f = relay.Function([data0, data1], glb_var(data0, data1))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        data1 = relay.var("data1", shape=shape, dtype=dtype)
        out = relay.add(data0, data1)
        main_f = relay.Function([data0, data1], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        return mod, ref_mod

    mod, ref_mod = gen_add()

    data0 = np.random.uniform(0, 1, shape).astype(dtype)
    data1 = np.random.uniform(0, 1, shape).astype(dtype)
    check_result(mod, ref_mod, {"data0": data0, "data1": data1}, shape, tol=1e-5)


def test_relu():
    """Test a subgraph with a single ReLU operator."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    shape = (1, 32, 14, 14)

    def gen_relu():
        data0 = relay.var("data0", shape=shape, dtype=dtype)
        out = relay.nn.relu(data0)

        func = relay.Function([data0], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        main_f = relay.Function([data0], glb_var(data0))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        data0 = relay.var("data0", shape=shape, dtype=dtype)
        out = relay.nn.relu(data0)
        main_f = relay.Function([data0], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        return mod, ref_mod

    mod, ref_mod = gen_relu()

    data0 = np.random.uniform(-1, 1, shape).astype(dtype)
    check_result(
        mod,
        ref_mod,
        {
            "data0": data0,
        },
        (1, 32, 14, 14),
        tol=1e-5,
    )


def test_dense():
    """Test a subgraph with a single dense operator."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    a_shape = (1, 512)
    b_shape = (1024, 512)

    def gen_dense():
        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        out = relay.nn.dense(a, b)

        func = relay.Function([a, b], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        main_f = relay.Function([a, b], glb_var(a, b))
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        a = relay.var("A", shape=a_shape, dtype=dtype)
        b = relay.var("B", shape=b_shape, dtype=dtype)
        out = relay.nn.dense(a, b)
        main_f = relay.Function([a, b], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        return mod, ref_mod

    mod, ref_mod = gen_dense()

    data_a = np.random.uniform(0, 1, a_shape).astype(dtype)
    data_b = np.random.uniform(0, 1, b_shape).astype(dtype)
    check_result(mod, ref_mod, {"A": data_a, "B": data_b}, (1, 1024), tol=1e-5)


def test_bn():
    """Test a subgraph with a single batch_norm operator."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    d_shape = (1, 8)
    c_shape = (8,)

    def gen_bn():
        data = relay.var("data", shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]

        func = relay.Function([data, gamma, beta, moving_mean, moving_var], out)
        func = set_func_attr(func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = func
        mod = transform.InferType()(mod)

        data = relay.var("data", shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        main_f = relay.Function(
            [data, gamma, beta, moving_mean, moving_var],
            glb_var(data, gamma, beta, moving_mean, moving_var),
        )
        mod["main"] = main_f
        mod = transform.InferType()(mod)

        data = relay.var("data", shape=d_shape)
        gamma = relay.var("gamma", shape=c_shape)
        beta = relay.var("beta", shape=c_shape)
        moving_mean = relay.var("moving_mean", shape=c_shape)
        moving_var = relay.var("moving_var", shape=c_shape)
        bn = relay.nn.batch_norm(data, gamma, beta, moving_mean, moving_var)
        out = bn[0]
        main_f = relay.Function([data, gamma, beta, moving_mean, moving_var], out)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_f
        ref_mod = transform.InferType()(ref_mod)

        return mod, ref_mod

    mod, ref_mod = gen_bn()

    data = np.random.uniform(-1, 1, d_shape).astype(dtype)
    gamma = np.random.uniform(-1, 1, c_shape).astype(dtype)
    beta = np.random.uniform(-1, 1, c_shape).astype(dtype)
    moving_mean = np.random.uniform(-1, 1, c_shape).astype(dtype)
    moving_var = np.random.uniform(-1, 1, c_shape).astype(dtype)
    check_result(
        mod,
        ref_mod,
        {
            "data": data,
            "gamma": gamma,
            "beta": beta,
            "moving_mean": moving_mean,
            "moving_var": moving_var,
        },
        d_shape,
        tol=1e-5,
    )


def test_multiple_ops():
    """Test a subgraph with multiple operators."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    w1shape = (32, 32, 3, 3)
    w2shape = (64, 32, 5, 5)

    def get_net():
        data = relay.var("data", relay.TensorType(ishape, dtype))
        w1 = relay.var("w1", relay.TensorType(w1shape, dtype))
        w2 = relay.var("w2", relay.TensorType(w2shape, dtype))

        layer = relay.nn.conv2d(data=data, weight=w1, kernel_size=(3, 3), padding=(1, 1))
        layer = relay.nn.relu(layer)
        layer = relay.nn.conv2d(data=layer, weight=w2, kernel_size=(5, 5), padding=(2, 2))
        layer = relay.nn.relu(layer)

        main_f = relay.Function([data, w1, w2], layer)
        mod = tvm.IRModule()
        mod["main"] = main_f
        return mod

    def get_partitoned_mod(mod):
        remove_bn_pass = tvm.transform.Sequential(
            [
                transform.InferType(),
                transform.SimplifyInference(),
                transform.FoldConstant(),
                transform.FoldScaleAxis(),
            ]
        )
        byoc_pass = tvm.transform.Sequential(
            [
                remove_bn_pass,
                transform.AnnotateTarget("dnnl"),
                transform.MergeCompilerRegions(),
                transform.PartitionGraph(),
            ]
        )

        with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
            return byoc_pass(mod)

    ref_mod = get_net()
    mod = get_partitoned_mod(ref_mod)

    data = np.random.uniform(0, 1, ishape).astype(dtype)
    w1 = np.random.uniform(0, 1, w1shape).astype(dtype)
    w2 = np.random.uniform(0, 1, w2shape).astype(dtype)
    check_result(
        mod,
        ref_mod,
        {
            "data": data,
            "w1": w1,
            "w2": w2,
        },
        (1, 64, 14, 14),
        tol=1e-5,
    )


def test_composite():
    """Test DNNL patterns and there composite functions."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"

    def conv2d_relu():
        ishape = (1, 32, 14, 14)
        w1shape = (32, 32, 3, 3)

        # Composite function
        in_1 = relay.var("in_1", shape=ishape, dtype=dtype)
        in_2 = relay.var("in_2", shape=w1shape, dtype=dtype)
        conv2d = relay.nn.conv2d(in_1, in_2, kernel_size=(3, 3), padding=(1, 1))
        relu = relay.nn.relu(conv2d)
        func = relay.Function([in_1, in_2], relu)
        func = func.with_attr("Composite", "dnnl.conv2d_relu")
        func = func.with_attr("PartitionedFromPattern", "nn.conv2d_nn.relu_")

        # Partition function
        arg_1 = relay.var("arg_1", shape=ishape, dtype=dtype)
        arg_2 = relay.var("arg_2", shape=w1shape, dtype=dtype)
        call = relay.Call(func, [arg_1, arg_2])
        p_func = relay.Function([arg_1, arg_2], call)
        p_func = set_func_attr(p_func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = p_func
        mod = transform.InferType()(mod)

        # Main function
        data = relay.var("data", shape=ishape, dtype=dtype)
        weight = relay.var("weight", shape=w1shape, dtype=dtype)
        main_func = relay.Function([data, weight], glb_var(data, weight))
        mod["main"] = main_func
        mod = transform.InferType()(mod)

        # Reference module
        data = relay.var("data", shape=ishape, dtype=dtype)
        weight = relay.var("weight", shape=w1shape, dtype=dtype)
        conv2d = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1))
        relu = relay.nn.relu(conv2d)
        main_func = relay.Function([data, weight], relu)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_func
        ref_mod = transform.InferType()(ref_mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w1_data}, (1, 32, 14, 14)

    def conv2d_bias_relu():
        ishape = (1, 32, 14, 14)
        w1shape = (32, 32, 3, 3)
        bshape = (32, 1, 1)

        # Composite function
        in_1 = relay.var("in_1", shape=ishape, dtype=dtype)
        in_2 = relay.var("in_2", shape=w1shape, dtype=dtype)
        in_3 = relay.var("in_3", shape=bshape, dtype=dtype)
        conv2d = relay.nn.conv2d(in_1, in_2, kernel_size=(3, 3), padding=(1, 1))
        add = relay.add(conv2d, in_3)
        relu = relay.nn.relu(add)
        func = relay.Function([in_1, in_2, in_3], relu)
        func = func.with_attr("Composite", "dnnl.conv2d_bias_relu")
        func = func.with_attr("PartitionedFromPattern", "nn.conv2d_add_nn.relu_")

        # Partition function
        arg_1 = relay.var("arg_1", shape=ishape, dtype=dtype)
        arg_2 = relay.var("arg_2", shape=w1shape, dtype=dtype)
        arg_3 = relay.var("arg_3", shape=bshape, dtype=dtype)
        call = relay.Call(func, [arg_1, arg_2, arg_3])
        p_func = relay.Function([arg_1, arg_2, arg_3], call)
        p_func = set_func_attr(p_func, "dnnl", "dnnl_0")
        glb_var = relay.GlobalVar("dnnl_0")
        mod = tvm.IRModule()
        mod[glb_var] = p_func
        mod = transform.InferType()(mod)

        # Main function
        data = relay.var("data", shape=ishape, dtype=dtype)
        weight = relay.var("weight", shape=w1shape, dtype=dtype)
        bias = relay.var("bias", shape=bshape, dtype=dtype)
        main_func = relay.Function([data, weight, bias], glb_var(data, weight, bias))
        mod["main"] = main_func
        mod = transform.InferType()(mod)

        # Reference module
        data = relay.var("data", shape=ishape, dtype=dtype)
        weight = relay.var("weight", shape=w1shape, dtype=dtype)
        bias = relay.var("bias", shape=bshape, dtype=dtype)
        conv2d = relay.nn.conv2d(data, weight, kernel_size=(3, 3), padding=(1, 1))
        add = relay.add(conv2d, bias)
        relu = relay.nn.relu(add)
        main_func = relay.Function([data, weight, bias], relu)
        ref_mod = tvm.IRModule()
        ref_mod["main"] = main_func
        ref_mod = transform.InferType()(ref_mod)

        i_data = np.random.uniform(0, 1, ishape).astype(dtype)
        w1_data = np.random.uniform(0, 1, w1shape).astype(dtype)
        b_data = np.random.uniform(0, 1, bshape).astype(dtype)

        return mod, ref_mod, {"data": i_data, "weight": w1_data, "bias": b_data}, (1, 32, 14, 14)

    for mod, ref_mod, input_maps, out_shape in [conv2d_relu(), conv2d_bias_relu()]:
        check_result(mod, ref_mod, input_maps, out_shape, tol=1e-5)


def test_constant():
    """Test the subgraph with (var, const, ...) arguments."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (1, 32, 14, 14)
    wshape = (32, 32, 3, 3)

    data = relay.var("data", shape=ishape, dtype=dtype)
    weight = relay.var("weight", shape=wshape, dtype=dtype)
    bn_gamma = relay.var("bn_gamma")
    bn_beta = relay.var("bn_beta")
    bn_mmean = relay.var("bn_mean")
    bn_mvar = relay.var("bn_var")

    layer = relay.nn.conv2d(data=data, weight=weight, kernel_size=(3, 3), padding=(1, 1))
    bn_output = relay.nn.batch_norm(layer, bn_gamma, bn_beta, bn_mmean, bn_mvar)
    out = bn_output[0]
    out = relay.nn.relu(out)

    func = relay.Function(relay.analysis.free_vars(out), out)
    ref_mod, params = tvm.relay.testing.create_workload(func)
    ref_mod["main"] = bind_params_by_name(ref_mod["main"], params)

    remove_bn_pass = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
        ]
    )

    dnnl_patterns = get_pattern_table("dnnl")
    composite_partition = tvm.transform.Sequential(
        [
            transform.MergeComposite(dnnl_patterns),
            transform.AnnotateTarget("dnnl"),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        ref_mod = remove_bn_pass(ref_mod)
        mod = composite_partition(ref_mod)

    i_data = np.random.uniform(0, 1, ishape).astype(dtype)
    check_result(mod, ref_mod, {"data": i_data}, (1, 32, 14, 14), tol=1e-5)


def test_partial_constant():
    """Test the subgraph with (const, var, const, var) arguments."""
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    dtype = "float32"
    ishape = (10, 10)

    in_1 = relay.var("in_1", shape=ishape, dtype=dtype)
    in_2 = relay.var("in_2", shape=ishape, dtype=dtype)
    in_3 = relay.var("in_3", shape=ishape, dtype=dtype)
    in_4 = relay.var("in_4", shape=ishape, dtype=dtype)

    add1 = relay.add(in_1, in_2)
    add2 = relay.add(add1, in_3)
    add3 = relay.add(add2, in_3)
    add4 = relay.add(add3, in_3)

    func = relay.Function([in_1, in_2, in_3, in_4], add4)
    ref_mod = tvm.IRModule.from_expr(func)
    ref_mod = relay.transform.InferType()(ref_mod)

    data1 = np.random.uniform(0, 1, ishape).astype(dtype)
    data3 = np.random.uniform(0, 1, ishape).astype(dtype)

    params = {
        "in_1": tvm.nd.array(data1, device=tvm.cpu(0)),
        "in_3": tvm.nd.array(data3, device=tvm.cpu(0)),
    }
    ref_mod["main"] = bind_params_by_name(ref_mod["main"], params)

    opt_pass = tvm.transform.Sequential(
        [
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3, disabled_pass=["AlterOpLayout"]):
        mod = opt_pass(ref_mod)

    data2 = np.random.uniform(0, 1, ishape).astype(dtype)
    data4 = np.random.uniform(0, 1, ishape).astype(dtype)
    check_result(mod, ref_mod, {"in_2": data2, "in_4": data4}, (10, 10), tol=1e-5)


def test_qnn_conv2d():
    """Test the subgraph with conv2d->add->requantize->relu
    Have to test:
        only int8 conv (u8i8 i8i8)
        with/without relu
        with/without rescale
        rescale [{oc}, scl] [{oc}, {oc}] [scl, {oc}]
        dst out u8 i8 i32 float32
        bias type u8 i8 i32 float32
        rescale non const
        bias nonconst
        weight nonconst
    """
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    for data_zp, out_zp in (
            (0, 0),
            (0, 88),
            (5, 0),
            (15, 25),
    ):

        dtype = "float32"
        IH, IW = 5, 5
        KH, KW = 3, 3
        IC = 16
        OC = 8
        d_shape = (1, IH, IW, IC)
        w_shape = (KH, KW, IC, OC)
        b_shape = (OC,)

        # NB! to have bit exact out we should take into account that
        # some implementation may use 16bit intermediate accumulator
        # It may lead to 16bit accumulator overflow and result mismatch
        # in compare with ref scoring with using i32 accumulator only.
        #
        # Example:
        # u8*i8 u8*i8 u8*i8 u8*i8
        #   |     |    |      |
        #   u8    u8   u8    u8
        #     \   /     \   /
        #      i16       i16
        #         \     /
        #           i32
        #
        # Not the same as:
        #   res = conv(src.astype(i32), wgh.astype(i32)
        #
        use_stored = False
        if use_stored:
            data_d = np.load("data_d.npy")
            # data_d = np.full(d_shape, 5).astype("uint8")
            data_w = np.load("data_w.npy")
            data_b = np.load("data_b.npy")
        else:
            data_d = np.random.uniform(0, 20, d_shape).astype("uint8")
            data_w = np.random.uniform(-10, 10, w_shape).astype("int8")
            data_b = np.random.uniform(-10, 10, b_shape).astype("int32")
            np.save("data_d", data_d)
            np.save("data_w", data_w)
            np.save("data_b", data_b)

        in_d = relay.var("in_1", shape=d_shape, dtype="uint8")
        in_w = relay.const(data_w, dtype="int8")
        in_b = relay.const(data_b, dtype="int32")
        # conv quantize args
        in_d_zp = relay.const(data_zp, dtype="int32")
        in_d_sc = relay.const(1/10, dtype=dtype)
        in_w_zp = relay.const(0.0, dtype="int32")
        in_w_sc = relay.const(1/10, dtype=dtype)
        # re quantize
        rq_in_zp = relay.const(0.0, dtype="int32")
        rq_in_sc = relay.const(np.random.uniform(0.01, 0.02, (OC,)).astype("float32"), dtype=dtype)
        rq_out_zp = relay.const(out_zp, dtype="int32")
        rq_out_sc = relay.const(1/10, dtype=dtype)

        op = in_d
        op = tvm.relay.nn.pad(op, ((0, 0), (1, 1), (1, 1), (0, 0)), pad_value=data_zp)
        op = tvm.relay.qnn.op.conv2d(op, in_w, in_d_zp, in_w_zp, in_d_sc, in_w_sc,
                                     kernel_size=(KH, KW), padding=(0, 0),
                                     channels=OC, out_dtype="int32",
                                     data_layout="NHWC", kernel_layout="HWIO")
        op = tvm.relay.add(op, in_b)
        op = tvm.relay.qnn.op.requantize(op, rq_in_sc, rq_in_zp, rq_out_sc, rq_out_zp, out_dtype="int32")
        op = tvm.relay.clip(op, a_min=0.0, a_max=255.0)
        op = tvm.relay.cast(op, dtype="uint8")

        func = relay.Function([in_d], op)
        ref_mod = tvm.IRModule.from_expr(func)
        ref_mod = relay.transform.InferType()(ref_mod)

        mod = partition_for_dnnl(ref_mod)

        # atol=1 means int values should match with +-1 tolerance
        check_result(mod, ref_mod, {"in_1": data_d}, (1, IH, IW, OC),
                     tol=1e-10, atol=1, dtype="uint8")


def test_qnn_conv2d_sum():
    """Test the subgraph with conv2d->add->requantize->relu
    Have to test:
        only int8 conv (u8i8 i8i8)
        with/without relu
        with/without rescale
        rescale [{oc}, scl] [{oc}, {oc}] [scl, {oc}]
        dst out u8 i8 i32 float32
        bias type u8 i8 i32 float32
        rescale non const
        bias nonconst
        weight nonconst
    """
    if not tvm.get_global_func("runtime.DNNLJSONRuntimeCreate", True):
        print("skip because DNNL codegen is not available")
        return

    for data_zp, out_zp in (
            (0, 0),
            # (0, 88),
            # (5, 0),
            # (15, 25),
    ):

        dtype = "float32"
        IH, IW = 5, 5
        KH, KW = 3, 3
        IC = 16
        OC = 8
        d_shape = (1, IH, IW, IC)
        w_shape = (KH, KW, IC, OC)
        b_shape = (OC,)

        # in1 5 : scl=0.2  zp=0      // 1.0 in float
        # W 2 : scl=0.1  zp=0 (+/-)  // result 16x9x0.2 = +/-28.8   scl=0.02  zp=50
        # B 5 : // 0.5 in float
        # rescale // in scl=0.02 zp=50 , out scl=0.1 zp = 10
        # in2 25 : scl=0.3 zp=15     // 3.0 in float
        #

        use_stored = False
        if use_stored:
            data_in1 = np.load("data_in1.npy")
            data_in2 = np.load("data_in2.npy")
            data_w = np.load("data_w.npy")
            data_b = np.load("data_b.npy")
        else:
            data_in1 = np.full(d_shape, 5).astype("uint8")
            data_in2 = np.full((1, 5, 5, 8), 25).astype("uint8")
            data_w = np.full(w_shape, 2).astype("int8")
            data_w[:, :, :, ::2] *= -1
            data_b = np.full(b_shape, 5).astype("int32")
            # data_in1 = np.random.uniform(0, 20, d_shape).astype("uint8")
            # data_in2 = np.random.uniform(0, 20, (1, 5, 5, 8)).astype("uint8")
            # data_w = np.random.uniform(-10, 10, w_shape).astype("int8")
            # data_b = np.random.uniform(-10, 10, b_shape).astype("int32")
            # np.save("data_in1", data_in1)
            # np.save("data_in2", data_in2)
            # np.save("data_w", data_w)
            # np.save("data_b", data_b)

        in_1 = relay.var("in_1", shape=d_shape, dtype="uint8")
        in_2 = relay.var("in_2", shape=(1, 5, 5, 8), dtype="uint8")
        in_w = relay.const(data_w, dtype="int8")
        in_b = relay.const(data_b, dtype="int32")
        # qnn conv quantize args
        in_d_zp = relay.const(0, dtype="int32")
        in_d_sc = relay.const(0.2, dtype=dtype)
        in_w_zp = relay.const(0, dtype="int32")
        in_w_sc = relay.const(0.1, dtype=dtype)
        # re quantize
        rq_in_zp = relay.const(0.0, dtype="int32")
        rq_in_sc = relay.const(0.02, dtype=dtype)
        rq_out_zp = relay.const(10, dtype="int32")
        rq_out_sc = relay.const(0.1, dtype=dtype)
        # qnn add quant args
        lhs_zp = relay.const(10, dtype="int32")
        lhs_scl = relay.const(0.1, dtype="float32")
        rhs_zp = relay.const(15, dtype="int32")
        rhs_scl = relay.const(0.3, dtype="float32")
        out_zp = relay.const(0.0, dtype="int32")
        out_scl = relay.const(0.2, dtype="float32")

        op = in_1
        op = tvm.relay.qnn.op.conv2d(op, in_w, in_d_zp, in_w_zp, in_d_sc, in_w_sc,
                                     kernel_size=(KH, KW), padding=(1, 1),
                                     channels=OC, out_dtype="int32",
                                     data_layout="NHWC", kernel_layout="HWIO")
        op = tvm.relay.add(op, in_b)
        op = tvm.relay.qnn.op.requantize(op, rq_in_sc, rq_in_zp, rq_out_sc, rq_out_zp, out_dtype="int32")
        op = tvm.relay.clip(op, a_min=0.0, a_max=255.0)
        op = tvm.relay.cast(op, dtype="uint8")
        op = tvm.relay.qnn.op.add(op, in_2, lhs_scl, lhs_zp, rhs_scl, rhs_zp, out_scl, out_zp)
        op = tvm.relay.clip(op, a_min=0.0, a_max=255.0)

        func = relay.Function([in_1, in_2], op)
        ref_mod = tvm.IRModule.from_expr(func)
        ref_mod = relay.transform.InferType()(ref_mod)

        mod = partition_for_dnnl(ref_mod)

        # atol=1 means int values should match with +-1 tolerance
        check_result(mod, ref_mod, {"in_1": data_in1, "in_2": data_in2}, (1, IH, IW, OC),
                     tol=1e-10, atol=2, dtype="uint8")


#  TODO(@apeskov): have to check
#    * non scalar wght_zp
#    * non zero data_zp
#    * with activation
#    * with sum
def test_qnn_dense():
    """
    :return:
    """

    for data_zp, out_zp in (
            (0, 0),
            (0, 88),
            (5, 0),
            (15, 25),
    ):
        dtype = "float32"
        NB = 1
        IC = 16
        OC = 8
        d_shape = (NB, IC)
        w_shape = (OC, IC)
        b_shape = (OC,)

        data_i = np.random.uniform(0, 30, d_shape).astype("uint8")
        data_w = np.random.uniform(-10, 10, w_shape).astype("int8")
        data_b = np.random.uniform(-10, 10, b_shape).astype("int32")

        in_d = relay.var("in", shape=d_shape, dtype="uint8")
        in_w = relay.const(data_w, dtype="int8")
        in_b = relay.const(data_b, dtype="int32")
        # qnn dense quantize args
        in_d_zp = relay.const(data_zp, dtype="int32")
        in_d_sc = relay.const(0.1, dtype=dtype)
        in_w_zp = relay.const(0, dtype="int32")
        in_w_sc = relay.const(0.12, dtype=dtype)
        # re quantize
        rq_in_zp = relay.const(0.0, dtype="int32")
        rq_in_sc = relay.const(0.02, dtype=dtype)
        rq_out_zp = relay.const(out_zp, dtype="int32")
        rq_out_sc = relay.const(0.1, dtype=dtype)

        op = in_d
        op = tvm.relay.qnn.op.dense(op, in_w, in_d_zp, in_w_zp, in_d_sc, in_w_sc,
                                    units=OC, out_dtype="int32")
        op = tvm.relay.add(op, in_b)
        op = tvm.relay.qnn.op.requantize(op, rq_in_sc, rq_in_zp, rq_out_sc, rq_out_zp, out_dtype="int32")
        op = tvm.relay.clip(op, a_min=0.0, a_max=255.0)
        op = tvm.relay.cast(op, dtype="uint8")

        func = relay.Function([in_d], op)
        ref_mod = tvm.IRModule.from_expr(func)
        ref_mod = relay.transform.InferType()(ref_mod)

        mod = partition_for_dnnl(ref_mod)

        # atol=1 means int values should match with +-1 tolerance
        check_result(mod, ref_mod, {"in": data_i}, (1, OC),
                     tol=1e-10, atol=1, dtype="uint8")


if __name__ == "__main__":
    test_conv2d()
    test_add()
    test_relu()
    test_dense()
    test_bn()
    test_multiple_ops()
    test_composite()
    test_constant()
    test_partial_constant()
    test_qnn_conv2d()
    test_qnn_conv2d_sum()
    test_qnn_dense()

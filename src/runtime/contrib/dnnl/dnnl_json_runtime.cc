/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/dnnl/dnnl_json_runtime.cc
 * \brief A simple JSON runtime for DNNL.
 */

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <string>
#include <vector>

#include "../json/json_node.h"
#include "../json/json_runtime.h"
#include "dnnl.hpp"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class DNNLJSONRuntime : public JSONRuntimeBase {
  using tag = dnnl::memory::format_tag;
  using dt = dnnl::memory::data_type;

 public:
  DNNLJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                  const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "dnnl_json"; }

  void Init(const Array<NDArray>& consts) override {
    // Setup constants entries for weights.
    SetupConstants(consts);

    BuildEngine();

    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
  }

  void Run() override {
    // Fill in the input buffers.
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto eid = EntryID(input_nodes_[i], 0);
      if (std::find(input_var_eid_.begin(), input_var_eid_.end(), eid) == input_var_eid_.end())
        continue;
      // TODO(@apeskov): check if entry_out_mem_[eid] exists
      // TODO(@comaniac): Support other data lengths.
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      write_to_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                           offset_in_bytes);
    }

    // Invoke the engine through intepreting the stream.
    for (size_t i = 0; i < net_.size(); ++i) {
      net_.at(i).execute(stream_, net_args_.at(i));
    }
    stream_.wait();

    // Read output buffers.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      size_t offset_in_bytes = entry_out_mem_[eid].second * 4;
      size_t buffer_size = GetDataSize(*data_entry_[eid]);
      read_from_dnnl_memory(data_entry_[eid]->data, entry_out_mem_[eid].first, buffer_size,
                            offset_in_bytes);
    }
  }

 private:
  // Build up the engine based on the input graph.
  void BuildEngine() {
    engine_ = dnnl::engine(dnnl::engine::kind::cpu, 0);
    stream_ = dnnl::stream(engine_);

    // Build subgraph engine.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        ICHECK_EQ(node.GetOpType(), "kernel");
        auto op_name = node.GetOpName();
        if ("nn.conv2d" == op_name) {
          Conv2d(nid);
        } else if ("dnnl.conv2d_relu" == op_name) {
          Conv2d(nid, true, false);
        } else if ("dnnl.conv2d_bias_relu" == op_name) {
          Conv2d(nid, true, true);
        } else if ("dnnl.qnn.conv2d_relu" == op_name) {
          QnnConv2d(nid);
        } else if ("nn.dense" == op_name) {
          Dense(nid);
        } else if ("nn.batch_norm" == op_name) {
          BatchNorm(nid);
        } else if ("nn.relu" == op_name) {
          Relu(nid);
        } else if ("add" == op_name) {
          Add(nid);
        } else {
          LOG(FATAL) << "Unsupported op: " << op_name;
        }
      }
    }
  }

  // Bind a JSON graph node entry to a DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory::desc mem_desc,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    if (entry_out_mem_.count(eid) == 0) {
      if (data_entry_[eid])
        return BindDNNLMemoryToConst(entry, mem_desc, offset);
      else
        return BindDNNLMemory(entry, dnnl::memory(mem_desc, engine_), offset);
    }
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemory(const JSONGraphNodeEntry& entry, dnnl::memory mem,
                              size_t offset = 0) {
    auto eid = EntryID(entry);
    // Since the DNNL memory has been created before calling this function, we assume the entry
    // has not yet been bound to the other DNNL memory; otherwise it may have memory leak.
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK(dltype.bits == 32 || dltype.bits == 8);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  // Bind a JSON graph node entry to a given DNNL memory.
  dnnl::memory BindDNNLMemoryToConst(const JSONGraphNodeEntry& entry, dnnl::memory::desc desc,
                                     size_t offset = 0) {
    auto eid = EntryID(entry);
    ICHECK_EQ(entry_out_mem_.count(eid), 0);

    // TODO(@comanic): Support other data types (i.e., int8).
    auto data_node = nodes_[entry.id_];
    auto dltype = data_node.GetOpDataType()[entry.index_];
    ICHECK(dltype.bits == 32 || dltype.bits == 8);

    auto dl_tensor = data_entry_[eid];
    ICHECK(dl_tensor);

    auto mem = dnnl::memory(desc, engine_, dl_tensor->data);

    entry_out_mem_[eid] = {mem, offset};
    return entry_out_mem_[eid].first;
  }

  void Conv2d(const size_t& nid, const bool has_relu = false, const bool has_bias = false) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
        IC = input_shape[1],                    // input channels
        IH = input_shape[2],                    // input height
        IW = input_shape[2],                    // input width
        OC = weight_shape[0],                   // output channels
        KH = weight_shape[2],                   // weight height
        KW = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::f32, tag::any);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::f32, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::f32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::f32, tag::nchw);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;
    if (has_relu) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);
    net_.push_back(conv);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::f32, tag::nchw});

    // Weight memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
    auto conv2d_weights_memory = BindDNNLMemory(
        weight_entry, {weights_dims, dt::f32, (groups > 1) ? tag::goihw : tag::oihw});

    // Bias memory.
    auto conv2d_bias_memory = dnnl::memory({bias_dims, dt::f32, tag::x}, engine_);
    if (has_bias) {
      auto bias_entry = node.GetInputs()[2];
      BindDNNLMemory(bias_entry, conv2d_bias_memory);
    } else {
      std::vector<float> bias(OC, 0);
      write_to_dnnl_memory(bias.data(), conv2d_bias_memory, OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  template<typename T>
  std::vector<T> get_values(const JSONGraphNodeEntry& entry, size_t size) {
    auto eid = EntryID(entry);
    auto dl_tensor = data_entry_[eid];
    ICHECK(dl_tensor->dtype.bits == sizeof(T)*8);
    ICHECK(std::is_floating_point<T>::value ? dl_tensor->dtype.code == kDLFloat : true);
    ICHECK((std::is_integral<T>::value && std::is_signed<T>::value)
               ? dl_tensor->dtype.code == kDLInt : true);
    ICHECK((std::is_integral<T>::value && std::is_unsigned<T>::value)
               ? dl_tensor->dtype.code == kDLUInt : true);
    const T* ptr = static_cast<T*>(dl_tensor->data);
    if (dl_tensor->ndim == 0) {
      // broadcast
      return std::vector<T>(size, *ptr);
    } else {
//      ICHECK(dl_tensor->shape[0] == size);
      return std::vector<T>(ptr, ptr + size);
    }
  };

  std::vector<int32_t> quasi_conv(std::vector<int32_t>data , std::vector<int8_t> weight,
                                  int KH, int KW, int IC, int OC) {
    const auto* w_ptr = weight.data();
    std::vector<int32_t> res (OC, 0);
    for (int khw = 0; khw < KH*KW; khw++)
      for (int ic = 0; ic < IC; ic++)
        for (int oc = 0; oc < OC; oc++)
          res[oc] += data[ic] * static_cast<int32_t>(*w_ptr++);
    return res;
  }

  std::vector<int32_t> calc_out_shift(const JSONGraphNode& node, int KH, int KW, int IC, int OC) {
    auto wght_entry = node.GetInputs()[1];
    auto bias_entry = node.GetInputs()[6];

    auto data_zero_point_entry = node.GetInputs()[2];
    auto input_scale_entry = node.GetInputs()[7];
    auto input_zero_point_entry = node.GetInputs()[8];
    auto output_scale_entry = node.GetInputs()[9];
    auto output_zero_point_entry = node.GetInputs()[10];

    // out_scl = rq_in_scl<f32>/rq_out_scl<f32>
    // shft_src = 0
    // shft_dst = rq_out_zp<i32> + rq_in_scl<f32>/rq_out_scl<f32> * (- zp_D - rq_in_zp<i32>)
    //   zp_D = conv(zp_A * QW)

    auto zp_A = get_values<int>(data_zero_point_entry, IC);

    auto rq_out_zp = get_values<int32_t>(output_zero_point_entry, OC);
    auto rq_in_scl = get_values<float>(input_scale_entry, OC);
    auto rq_out_scl = get_values<float>(output_scale_entry, OC);
    auto rq_in_zp = get_values<int32_t>(input_zero_point_entry, OC);
    auto bias = get_values<int32_t>(bias_entry, OC);
    auto weight = get_values<int8_t>(wght_entry, KH*KW*IC*OC);

    auto zp_D = quasi_conv(zp_A, weight, KH, KW, IC, OC);

    std::vector<int32_t> res (OC, 0);

    for (int i = 0; i < OC; i++) {
      res[i] = rq_out_zp[i] + static_cast<int32_t>(rq_in_scl[i]/rq_out_scl[i] * (/*bias[i]*/ - rq_in_zp[i] - zp_D[i]));
    }
    return res;
  }

  std::vector<float> calc_out_scale(const JSONGraphNode& node, int KH, int KW, int IC, int OC) {
    auto input_scale_entry = node.GetInputs()[7];
    auto output_scale_entry = node.GetInputs()[9];

    auto rq_in_scl = get_values<float>(input_scale_entry, OC);
    auto rq_out_scl = get_values<float>(output_scale_entry, OC);

    std::vector<float> res (OC, 0);
    for (int i = 0; i < OC; i++) {
      res[i] = rq_in_scl[i]/rq_out_scl[i];
    }
    return res;
  }


  void QnnConv2d(const size_t& nid) {
    auto node = nodes_[nid];

    bool has_relu = false;
    bool has_bias = true;

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];
    std::vector<std::string> str_strides = node.GetAttr<std::vector<std::string>>("strides");
    std::vector<std::string> str_padding = node.GetAttr<std::vector<std::string>>("padding");
    dnnl::memory::dim groups = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);

    dnnl::memory::dim N = input_shape[0],       // batch size
        IH = input_shape[1],                    // input channels
        IW = input_shape[2],                    // input height
        IC = input_shape[3],                    // input width
        KH = weight_shape[0],                   // output channels
        KW = weight_shape[1],                   // weight height
        OC = weight_shape[3],                   // weight width
        PH_L = std::stoi(str_padding[1]),       // height padding: left
        PH_R = std::stoi(str_padding[3]),       // height padding: right
        PW_L = std::stoi(str_padding[0]),       // width padding: left
        PW_R = std::stoi(str_padding[2]),       // width padding: right
        SH = std::stoi(str_strides[0]),         // height-wise stride
        SW = std::stoi(str_strides[0]),         // weight-wise stride
        OH = (IH - KH + PH_L + PH_R) / SH + 1,  // output height
        OW = (IW - KW + PW_L + PW_R) / SW + 1;  // output width

    // Memory shapes.
    dnnl::memory::dims src_dims = {N, IC, IH, IW};
    dnnl::memory::dims weights_dims = {OC, IC, KH, KW};
    if (groups > 1) {
      weights_dims = {groups, 1, IC / groups, KH, KW};
    }
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims dst_dims = {N, OC, OH, OW};
    dnnl::memory::dims strides_dims = {SH, SW};
    dnnl::memory::dims padding_dims_l = {PH_L, PW_L};
    dnnl::memory::dims padding_dims_r = {PH_R, PW_R};

    // Memory descriptions.
    auto conv_src_md = dnnl::memory::desc(src_dims, dt::u8, tag::nhwc);
    auto conv_weights_md = dnnl::memory::desc(weights_dims, dt::s8, tag::any);
    auto conv_bias_md = dnnl::memory::desc(bias_dims, dt::s32, tag::any);
    auto conv_dst_md = dnnl::memory::desc(dst_dims, dt::u8, tag::nhwc);

    // Covn2d description.
    auto conv_desc = dnnl::convolution_forward::desc(
        dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct, conv_src_md,
        conv_weights_md, conv_bias_md, conv_dst_md, strides_dims, padding_dims_l, padding_dims_r);

    // Enable ReLU
    dnnl::primitive_attr attr;
    if (has_relu) {
      dnnl::post_ops ops;
      ops.append_eltwise(1.f, dnnl::algorithm::eltwise_relu, 0.f, 0.f);
      attr.set_post_ops(ops);
    }

    // requantize scale
    // 0-1  : data weight
    // 2-5  : data_zero_point, weight_zero_point, data_scale, weight_scale,
    // 6 : bias
    // 7-10 : input_scale, input_zero_point, output_scale, output_zero_point
    // output rescale
    auto out_scale = calc_out_scale(node, KH, KW, IC, OC);
    attr.set_output_scales(1 << 1, out_scale);

    // out zero point
    auto out_zero_point = calc_out_shift(node, KH, KW, IC, OC);
    dnnl::memory out_zp_mem({{OC}, dt::s32, tag::a}, engine_);
    std::copy(out_zero_point.begin(), out_zero_point.end(),
              static_cast<int32_t*>(out_zp_mem.get_data_handle()));
    attr.set_zero_points(DNNL_ARG_DST, 1<<1, {DNNL_RUNTIME_S32_VAL});

    auto conv2d_prim_desc = dnnl::convolution_forward::primitive_desc(conv_desc, attr, engine_);

    // Push to the network.
    auto conv = dnnl::convolution_forward(conv2d_prim_desc);

    // Data memory.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("data_layout")[0], "NHWC");
    auto conv2d_src_memory = BindDNNLMemory(data_entry, {src_dims, dt::u8, tag::nhwc});

    // Weight memory is in constants.
    ICHECK_EQ(node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "HWIO");
    auto weights_memory =
        BindDNNLMemory(weight_entry, {weights_dims, dt::s8, (groups > 1) ? tag::hwigo : tag::hwio});

    auto suggested_wgh_desc = conv2d_prim_desc.weights_desc();
    dnnl::memory conv2d_weights_memory = dnnl::memory(suggested_wgh_desc, engine_);
    auto w_reorder = dnnl::reorder(weights_memory, conv2d_weights_memory);
    w_reorder.execute(stream_, weights_memory, conv2d_weights_memory);

    // Bias memory.
    dnnl::memory conv2d_bias_memory;
    if (has_bias) {
      auto bias_entry = node.GetInputs()[6];
      conv2d_bias_memory = BindDNNLMemory(bias_entry, {bias_dims, dt::s32, tag::x});
    } else {
      std::vector<float> bias(OC, 0);
      write_to_dnnl_memory(bias.data(), conv2d_bias_memory, OC * sizeof(float));
    }

    // Output memory.
    JSONGraphNodeEntry out_entry(nid, 0);
    auto conv2d_dst_memory = BindDNNLMemory(out_entry, conv2d_prim_desc.dst_desc());

    // Bind memory buffers.
    net_.push_back(conv);
    net_args_.push_back({{DNNL_ARG_SRC, conv2d_src_memory},
                         {DNNL_ARG_WEIGHTS, conv2d_weights_memory},
                         {DNNL_ARG_BIAS, conv2d_bias_memory},
                         {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, out_zp_mem},
                         {DNNL_ARG_DST, conv2d_dst_memory}});
  }

  void Dense(const size_t& nid) {
    auto node = nodes_[nid];

    // Setup attributes.
    auto data_entry = node.GetInputs()[0];
    auto weight_entry = node.GetInputs()[1];
    dnnl::memory::dims input_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dims weight_shape = nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

    dnnl::memory::dim B = input_shape[0],  // batch size
        IC = input_shape[1],               // input channels
        OC = weight_shape[0];              // output channels

    // Memory shapes.
    dnnl::memory::dims data_dims = {B, IC};
    dnnl::memory::dims weight_dims = {OC, IC};
    dnnl::memory::dims bias_dims = {OC};
    dnnl::memory::dims out_dims = {B, OC};

    // Memory descriptions.
    auto data_md = dnnl::memory::desc({data_dims, dt::f32, tag::nc});
    auto weight_md = dnnl::memory::desc({weight_dims, dt::f32, tag::nc});
    auto bias_md = dnnl::memory::desc({bias_dims, dt::f32, tag::x});
    auto dst_md = dnnl::memory::desc({out_dims, dt::f32, tag::nc});

    // Dense description.
    auto dense_desc = dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference, data_md,
                                                        weight_md, bias_md, dst_md);
    auto dense_prim_desc = dnnl::inner_product_forward::primitive_desc(dense_desc, engine_);

    auto dense = dnnl::inner_product_forward(dense_prim_desc);
    net_.push_back(dense);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto weight_memory = BindDNNLMemory(weight_entry, weight_md);
    auto bias_memory = dnnl::memory(bias_md, engine_);
    std::vector<float> bias(OC, 0);
    write_to_dnnl_memory(bias.data(), bias_memory, OC * sizeof(float));
    JSONGraphNodeEntry out_entry(nid, 0);
    auto dst_memory = BindDNNLMemory(out_entry, dense_prim_desc.dst_desc());

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_WEIGHTS, weight_memory},
                         {DNNL_ARG_BIAS, bias_memory},
                         {DNNL_ARG_DST, dst_memory}});
  }

  void BatchNorm(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    auto gamma_entry = node.GetInputs()[1];
    auto beta_entry = node.GetInputs()[2];
    auto mean_entry = node.GetInputs()[3];
    auto variance_entry = node.GetInputs()[4];
    dnnl::memory::dims data_shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    dnnl::memory::dim IC = data_shape[1];
    float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);

    // Memory description.
    dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

    // BN description.
    auto bn_desc = dnnl::batch_normalization_forward::desc(
        dnnl::prop_kind::forward_inference, data_md, epsilon,
        dnnl::normalization_flags::use_global_stats | dnnl::normalization_flags::use_scale_shift);
    auto bn_prim_desc = dnnl::batch_normalization_forward::primitive_desc(bn_desc, engine_);
    auto bn = dnnl::batch_normalization_forward(bn_prim_desc);
    net_.push_back(bn);

    // Memories.
    auto data_memory = BindDNNLMemory(data_entry, data_md);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, data_md);
    auto mean_memory = BindDNNLMemory(mean_entry, bn_prim_desc.mean_desc());
    auto variance_memory = BindDNNLMemory(variance_entry, bn_prim_desc.variance_desc());

    // In DNNL, weight is composed of gamma+beta, so we point them to the same DNNL memory but
    // assign an offset to beta data for runtime serialization.
    auto weight_memory = BindDNNLMemory(gamma_entry, bn_prim_desc.weights_desc(), 0);
    BindDNNLMemory(beta_entry, weight_memory, IC);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory},
                         {DNNL_ARG_DST, out_memory},
                         {DNNL_ARG_SCALE_SHIFT, weight_memory},
                         {DNNL_ARG_MEAN, mean_memory},
                         {DNNL_ARG_VARIANCE, variance_memory}});
  }

  void Relu(const size_t& nid) {
    auto node = nodes_[nid];

    auto data_entry = node.GetInputs()[0];
    dnnl::memory::dims shape = nodes_[data_entry.id_].GetOpShape()[data_entry.index_];
    auto data_md = dnnl::memory::desc{{shape}, dt::f32, tag::abcd};

    auto relu_desc = dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                                 dnnl::algorithm::eltwise_relu, data_md, 0);
    auto relu_prim_desc = dnnl::eltwise_forward::primitive_desc(relu_desc, engine_);
    ICHECK(data_md == relu_prim_desc.dst_desc());

    auto relu = dnnl::eltwise_forward(relu_prim_desc);
    net_.push_back(relu);

    auto data_memory = BindDNNLMemory(data_entry, data_md);
    auto out_md = dnnl::memory::desc(shape, dt::f32, tag::abcd);
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    net_args_.push_back({{DNNL_ARG_SRC, data_memory}, {DNNL_ARG_DST, out_memory}});
  }

  void Add(const size_t& nid) {
    auto node = nodes_[nid];

    // Memory and compute description.
    std::vector<dnnl::memory::dims> data_dims;
    std::vector<dnnl::memory::desc> data_mds;
    std::vector<dnnl::memory> data_memories;

    ICHECK_EQ(node.GetInputs().size(), 2U);
    for (auto entry : node.GetInputs()) {
      auto data_shape = nodes_[entry.id_].GetOpShape()[entry.index_];
      dnnl::memory::desc data_md = GenDNNLMemDescByShape(data_shape, dt::f32);

      data_dims.push_back(data_shape);
      data_mds.push_back(data_md);
      data_memories.push_back(BindDNNLMemory(entry, data_md));
    }
    ICHECK(data_dims[0] == data_dims[1]);
    auto out_md = data_mds[0];
    JSONGraphNodeEntry out_entry(nid, 0);
    auto out_memory = BindDNNLMemory(out_entry, out_md);

    auto add_desc =
        dnnl::binary::desc(dnnl::algorithm::binary_add, data_mds[0], data_mds[1], out_md);
    auto add_prim_desc = dnnl::binary::primitive_desc(add_desc, engine_);
    auto add = dnnl::binary(add_prim_desc);
    net_.push_back(add);

    net_args_.push_back({{DNNL_ARG_SRC_0, data_memories[0]},
                         {DNNL_ARG_SRC_1, data_memories[1]},
                         {DNNL_ARG_DST, out_memory}});
  }

  // Read from DNNL memory (+offset) and write to the handle.
  inline void read_from_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                    size_t offset = 0) {
    ICHECK(mem);
    ICHECK(mem.get_desc().get_size() == size)  << "Wrong mem size";
    uint8_t* src = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(src + offset, src + offset + size, static_cast<uint8_t*>(handle));
  }

  // Read from the handle and write to DNNL memory (+offset).
  inline void write_to_dnnl_memory(void* handle, const dnnl::memory& mem, size_t size,
                                   size_t offset = 0) {
    ICHECK(mem);
    ICHECK(mem.get_desc().get_size() == size)  << "Wrong mem size";
    uint8_t* dst = static_cast<uint8_t*>(mem.get_data_handle());
    std::copy(reinterpret_cast<uint8_t*>(handle), reinterpret_cast<uint8_t*>(handle) + size,
              dst + offset);
  }

  // Generate DNNL memory description and infer the data layout by the given shape.
  inline dnnl::memory::desc GenDNNLMemDescByShape(const dnnl::memory::dims& shape, dt dtype) {
    dnnl::memory::desc data_md;
    switch (shape.size()) {
      case 2:
        data_md = dnnl::memory::desc({shape, dtype, tag::ab});
        break;
      case 3:
        data_md = dnnl::memory::desc({shape, dtype, tag::abc});
        break;
      case 4:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcd});
        break;
      case 5:
        data_md = dnnl::memory::desc({shape, dtype, tag::abcde});
        break;
      default:
        LOG(FATAL) << "Unsupported data shape dimension: " << shape.size();
        break;
    }
    return data_md;
  }

  /* The dnnl engine. */
  dnnl::engine engine_;
  /* The dnnl stream. */
  dnnl::stream stream_;
  /* The network layers that are represented in dnnl primitives. */
  std::vector<dnnl::primitive> net_;
  /* The memory that is consumed by arguments. */
  std::vector<std::unordered_map<int, dnnl::memory>> net_args_;
  /* The entry ID to its corresponding output memory. */
  std::unordered_map<uint32_t, std::pair<dnnl::memory, size_t>> entry_out_mem_;
};

runtime::Module DNNLJSONRuntimeCreate(String symbol_name, String graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<DNNLJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.DNNLJSONRuntimeCreate").set_body_typed(DNNLJSONRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_dnnl_json")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<DNNLJSONRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

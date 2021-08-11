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
#include "tvm/runtime/module.h"
#include "tvm/runtime/registry.h"

#include <numeric>

namespace demo {

using namespace tvm::runtime;

size_t getNumElements(NDArray arr) {
  auto shape = arr.Shape();
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<ShapeTuple::index_type>());
}

size_t getSizeInBytes(NDArray arr) {
  return getNumElements(arr) * arr.DataType().bytes();
}

class AsyncLauncherFactoryNode : public ModuleNode {
 public:
  explicit AsyncLauncherFactoryNode(Module mod) : lib_mod_(mod) {}

  virtual const char* type_key() const final { return "AsyncLauncherFactory"; }

  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "init") {
      return TypedPackedFunc<void(int)>([sptr_to_self, this](int worker_id) {
        return this->init_(worker_id);
      });
    } else if (name == "infer") {
      return TypedPackedFunc<NDArray(NDArray)>([sptr_to_self, this](NDArray input) {
        return this->infer_(input);
      });
    } else {
      LOG(FATAL) << "unknown function " << name;
      return PackedFunc();
    }
  }

 private:
  void init_(int worker_id) {
    graph_executor_ = lib_mod_.GetFunction("default")(dev_);
    run_ = graph_executor_.GetFunction("run");
    set_input_zero_copy_ = graph_executor_.GetFunction("set_input_zero_copy");
    get_input_ = graph_executor_.GetFunction("get_input");
    get_output_ = graph_executor_.GetFunction("get_output");
    get_num_outputs_ = graph_executor_.GetFunction("get_num_outputs");
    get_num_inputs_ = graph_executor_.GetFunction("get_num_inputs");

    // Only classification cases. One input, one output.
    ICHECK(static_cast<int>(get_num_outputs_()) == 1);
    ICHECK(static_cast<int>(get_num_inputs_()) == 1);

    NDArray input = get_input_(0);
    input_shape_ = input.Shape();

    NDArray output = get_output_(0);
    output_shape_ = output.Shape();

    // Assumption that Batch is first dimension of input and output tensor
    ICHECK_EQ(input_shape_[0], output_shape_[0]);
    model_batch_ = input_shape_[0];
  }

  NDArray infer_(NDArray input) {
    ICHECK_GE(input.Shape().size(), 1);
    auto cur_batch = input.Shape()[0];
    auto num_chunk = (cur_batch - 1) / model_batch_ + 1;

    std::vector<ShapeTuple::index_type> output_shape {output_shape_.begin(), output_shape_.end()};
    output_shape[0] = cur_batch;
    NDArray output = NDArray::Empty(output_shape, input.DataType(), dev_);

    for (size_t chunk_idx = 0; chunk_idx < num_chunk; chunk_idx++) {
      // Inplace chunk view on input/output
      NDArray batch_chunk_in = get_batch_chunk(input, model_batch_, chunk_idx);
      NDArray batch_chunk_out = get_batch_chunk(output, model_batch_, chunk_idx);

      set_input_zero_copy_(0, batch_chunk_in);
      run_();
      NDArray chunk_res = get_output_(0);
      chunk_res.CopyTo(batch_chunk_out);
    }
    return output;
  }

  static NDArray get_batch_chunk(NDArray src, size_t chunk_batch_size, size_t chunk_idx) {
    ICHECK(src.IsContiguous()) << "Strides support is no implemented";
    ICHECK_GE(src.Shape()[0], chunk_batch_size * (chunk_idx + 1))
        << "Auto zeroing is not supported";

    std::vector<ShapeTuple::index_type> chunk_dims {src.Shape().begin(), src.Shape().end()};
    chunk_dims[0] = chunk_batch_size;

    NDArray chunk = src.CreateView(chunk_dims, src.DataType());

    // WA. No easy way to specify new offset during CreateView call
    //     Also "set_input_zero_copy" doesn't support tensors with offset.
    const_cast<DLTensor*>(chunk.operator->())->data = static_cast<uint8_t*>(chunk->data) +
        getSizeInBytes(chunk) * chunk_idx;

    return chunk;
  }

 private:
  DLDevice dev_ {kDLCPU, 0};
  Module lib_mod_;
  Module graph_executor_;
  PackedFunc run_;
  PackedFunc set_input_zero_copy_;
  PackedFunc get_input_;
  PackedFunc get_output_;
  PackedFunc get_num_outputs_;
  PackedFunc get_num_inputs_;

  ShapeTuple::index_type model_batch_;

  ShapeTuple input_shape_;
  ShapeTuple output_shape_;
};

void CreateAsyncLauncherFactoryModule_(TVMArgs args, TVMRetValue* rv) {
  Module lib_mod = args[0];
  *rv = Module(make_object<AsyncLauncherFactoryNode>(lib_mod));
}

void SetAffinity_(TVMArgs args, TVMRetValue* rv) {
  int numa_id = args[0];
  int core_start = args[0];
  int core_end = args[0];
  std::cout << "[STUB] try to bind to NUMA_" << numa_id
            << " cores [" << core_start << ", " << core_end << "]"
            << std::endl;
}

TVM_REGISTER_GLOBAL("tvm.async_executor.create").set_body(demo::CreateAsyncLauncherFactoryModule_);
TVM_REGISTER_GLOBAL("tvm.async_executor.set_affinity").set_body(demo::SetAffinity_);


//TVM_DLL_EXPORT_PACKED_FUNC(CreateAsyncLauncher, demo::CreateAsyncLauncherFactoryModule_);
//TVM_DLL_EXPORT_PACKED_FUNC(SetAffinity_, demo::SetAffinity_);
}  // namespace demo

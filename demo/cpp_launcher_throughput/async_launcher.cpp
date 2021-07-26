#include <iostream>

#include "tvm/runtime/module.h"

namespace demo {

using namespace tvm::runtime;

class AsyncLauncherFactoryNode : public ModuleNode {
 public:
  explicit AsyncLauncherFactoryNode(int value) : value_(value) {}

  virtual const char* type_key() const final { return "AsyncLauncherFactory"; }

  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "add") {
      return TypedPackedFunc<int(int)>([sptr_to_self, this](int value) { return value_ + value; });
    } else if (name == "mul") {
      return TypedPackedFunc<int(int)>([sptr_to_self, this](int value) { return value_ * value; });
    } else {
      LOG(FATAL) << "unknown function " << name;
      return PackedFunc();
    }
  }

 private:
  int value_;
};

void CreateAsyncLauncherFactoryModule_(TVMArgs args, TVMRetValue* rv) {
  int value
      = args[0];
  *rv = Module(make_object<AsyncLauncherFactoryNode>(value));
}

// Use TVM_EXPORT_PACKED_FUNC to export a function with
TVM_DLL_EXPORT_PACKED_FUNC(CreateMyModule, demo::CreateAsyncLauncherFactoryModule_);
}  // namespace demo




//int main () {
//  std::string model_path = "/Users/apeskov/git/tvm/demo/__prebuilt/dnnl_int8_resnet50.so";
//  auto mod_factory = tvm::runtime::Module::LoadFromFile(model_path);
//
//  auto params = mod_factory.GetFunction("get_params")();
//
//  DLDevice dev{kDLCPU, 0};
//  // create the graph runtime module
//  tvm::runtime::Module gmod1 = mod_factory.GetFunction("default")(dev);
////  tvm::runtime::Module gmod2 = mod_factory.GetFunction("default")(dev);
////  tvm::runtime::Module gmod3 = mod_factory.GetFunction("default")(dev);
//
//
//  tvm::runtime::PackedFunc run = gmod1.GetFunction("run");
//  tvm::runtime::PackedFunc set_input = gmod1.GetFunction("set_input");
//  tvm::runtime::PackedFunc get_output = gmod1.GetFunction("get_output");
//
////  tvm::runtime::PackedFunc share_params = gmod2.GetFunction("share_params");
////  share_params(gmod1, params);
////  Time::time_point time4 = Time::now();
//  run();
////  Time::time_point time5 = Time::now();
//
//  std::cout << "Hello" << std::endl;
//  return 0;
//}

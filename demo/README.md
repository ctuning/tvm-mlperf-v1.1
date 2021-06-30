DNNL int8 scoring demo (Resnet-50)
==================================

## TVM build prerequisites:
1. Checkout DNNL by tag v2.2.3.
   
2. Build DNNL with OpenMP. It's a default threading runtime, but you have 
   to check if it was detected properly. LLVM Clang should support OMP.
   
   ```shell
   CC=/usr/local/Cellar/llvm/11.1.0_1/bin/clang \
   CXX=/usr/local/Cellar/llvm/11.1.0_1/bin/clang++ \
   cmake .. \ 
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=bin
   ```
   
3. Build TVM with OpenMP and DNNL support. OMP runtime should be the same
   
   ```shell
   -DUSE_LLVM=/usr/local/Cellar/llvm/11.1.0_1/bin/llvm-config
   -DUSE_GRAPH_EXECUTOR_DEBUG=ON
   -DUSE_DNNL_CODEGEN=ON
   -DCMAKE_PREFIX_PATH=<path-to-dnnl>/__build/bin
   -DUSE_OPENMP=gnu
   ```

## Demo script
Download quantized pytorch model(Resnet50), and pass is to script.
```shell
OMP_NUM_THREADS=8 OMP_PROC_BIND=true python ./int8_resnet_demo.py <path-to-pytorch-model> <path-to-cat-image> 
```
I recommend to limiting num of threads via OMP_NUM_THREADS=<number-of-cores>
(like a disabling of hyper-threading) and bind it to cores.  

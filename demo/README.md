DNNL int8 scoring demo (Resnet-50)
==================================

## TVM build prerequisites

1. Use TVM with support of DNNL-int8. Currently, this is a dev branch:
   https://github.com/apeskov/tvm/tree/ap/dnnl-int8

2. Use DNNL by tag v2.2.4.
   
3. Build DNNL with OpenMP. It's a default threading runtime, but you have
   to check if it was detected properly. LLVM Clang should support OMP.
   
   ```shell
   CC=/usr/local/Cellar/llvm/11.1.0_1/bin/clang \
   CXX=/usr/local/Cellar/llvm/11.1.0_1/bin/clang++ \
   cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=bin
   ```
   
4. Build TVM with OpenMP and DNNL support. OMP runtime should be the same
   
   ```shell
   -DUSE_LLVM=/usr/local/Cellar/llvm/11.1.0_1/bin/llvm-config
   -DUSE_GRAPH_EXECUTOR_DEBUG=ON
   -DUSE_DNNL_CODEGEN=ON
   -DCMAKE_PREFIX_PATH=<path-to-dnnl>/__build/bin
   -DUSE_OPENMP=gnu
   ```

5. Download model https://zenodo.org/record/4589637/files/resnet50_INT8bit_quantized.pt and put it into __data folder.

**Issue**: There is a compatibility problem with pyTorch package. Pytorch is also contains libomp.so library. So if
we will use torch and tvm packages simultaniously it will lead to crash inside omp runtime. As a WA we are using none
OpenMP version of TVM to compile pytorch model.


## Docker wrapper

For convenience of building and model compilation there is a docker equivalent which prepare environment properly.

Example of commands to prepare image and run commend inside container:
   ```shell
      cd <root-of-tvm>
      docker build . --tag tvm_dnnl_int8 -f ./demo/docker/Dockerfile.dnnl_build
      docker run -tdi -v ${PWD}/demo:/workspace --name int8_bench tvm_dnnl_int8 bash
      docker attach int8_bench
   ```

Inside of container you may:
1. Run compiler script which also will check accuracy comparing scoring results with original torch output.
   The resulting compiled model is in '__prebuilt' folder.
   ```shell
     cd /workspace; LD_LIBRARY_PATH=${TVM_LIBS} python3 int8_resnet50_demo_compile.py
   ```

2. Run performance checker. It supports Latency and Throughput mode of operation. Performance results are in ms.
   Metrics will be printed on screen and also dumped into csv format.
   ```shell
      cd /workspace; LD_LIBRARY_PATH=${TVM_LIBS_OMP} python3 int8_resnet50_demo_throughput.py
   ```
3. Prepare portable package to transfer it into some other linux machine and perform performance testing on it.
   (as example testing with cloud VMs). It will produce tgz archive with int8_resnet50_demo_throughput.py script inside.
   On target machine you will have to only install prerequisites and run script.

   NB!!! Compilation step should be completed before. Relocatable package should contain precompiled model.

   ```shell
      cd /workspace; ./prepare_package.sh
   ```


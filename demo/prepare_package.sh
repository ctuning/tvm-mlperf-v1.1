#!/usr/bin/env bash

mkdir /tmp/demo_pack
mkdir /tmp/demo_pack/lib
mkdir /tmp/demo_pack/python

cp /tmp/exp/__build_tvm_omp/libtvm_runtime.so /tmp/demo_pack/lib
cp /tmp/exp/__build_tvm_omp/libtvm.so /tmp/demo_pack/lib
cp /tmp/exp/__build_dnnl_omp/bin/lib/libdnnl.so.2 /tmp/demo_pack/lib
cp -r /tmp/exp/tvm/python /tmp/demo_pack/.

cp -r __data /tmp/demo_pack/.
cp -r __prebuilt /tmp/demo_pack/.

cp int8_resnet50_demo_throughput.py /tmp/demo_pack/.
echo "Pillow \nnumpy \ndecorator \nscipy \nattrs \npytest" > /tmp/demo_pack/requirements.txt
echo "sudo apt-get install llvm-12 clang-12 python3-pip\n pip3 install -r ./requirements.txt" > /tmp/demo_pack/prerequisite.sh

tar -czf demo_pack.tgz --exclude __pycache__ -C /tmp demo_pack
rm -rf /tmp/demo_pack

echo "Result: demo_pack.tgz"

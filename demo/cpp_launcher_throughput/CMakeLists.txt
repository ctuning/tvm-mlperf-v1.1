cmake_minimum_required(VERSION 3.2)
project(tvm C CXX)

add_library(async_launcher SHARED async_launcher.cpp)
target_link_libraries(async_launcher tvm)

#if (NOT TARGET tvm_runtime)
#    # Build outside of tvm source tree
#    add_library(tvm_runtime SHARED IMPORTED)
#    set_target_properties(tvm_runtime PROPERTIES
#            INTERFACE_INCLUDE_DIRECTORIES "${TVM_ROOT}/include ${TVM_ROOT}/3rdparty/dmlc-core/include"
##            INTERFACE_LINK_LIBRARIES ""
#            IMPORTED_LOCATION "${TVM_BUILD}/build"
#            )
#endif()

find_package(Threads REQUIRED)

#link_directories(${TVM_BUILD})
#include_directories(${TVM_ROOT}/include ${TVM_ROOT}/3rdparty/dmlc-core/include ${TVM_ROOT}/3rdparty/dlpack/include)
add_executable(async_launcher_demo main.cpp)
target_link_libraries(async_launcher_demo tvm Threads::Threads)

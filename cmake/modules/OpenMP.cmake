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

if(APPLE AND NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
  link_directories("/usr/local/Cellar/llvm/11.1.0_1/lib")
  if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    set(OpenMP_C "${CMAKE_C_COMPILER}")
    set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
    set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
  endif()
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
    set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
  endif()
endif()

# OpenMP Module
if(USE_OPENMP STREQUAL "gnu")
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#    list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenMP_CXX_LIBRARIES})
    add_definitions(-DTVM_THREADPOOL_USE_OPENMP=1)
    message(STATUS "Build with OpenMP ${OpenMP_CXX_LIBRARIES}")
  else()
    add_definitions(-DTVM_THREADPOOL_USE_OPENMP=0)
    message(WARNING "OpenMP cannot be found, use TVM threadpool instead.")
  endif()
elseif(USE_OPENMP STREQUAL "intel")
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    if (MSVC)
      find_library(OMP_LIBRARY NAMES libiomp5md)
    else()
      find_library(OMP_LIBRARY NAMES iomp5)
    endif()
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${OMP_LIBRARY})
    add_definitions(-DTVM_THREADPOOL_USE_OPENMP=1)
    message(STATUS "Build with OpenMP " ${OMP_LIBRARY})
  else()
    add_definitions(-DTVM_THREADPOOL_USE_OPENMP=0)
    message(WARNING "OpenMP cannot be found, use TVM threadpool instead.")
  endif()
else()
  add_definitions(-DTVM_THREADPOOL_USE_OPENMP=0)
endif()

#! /usr/bin/env bash
# Copyright 2020-2022 Autoware Foundation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

TVM_VERSION="v0.10.0"
TVM_BASE_DIR="/tmp/tvm"
TVM_BUILD_DIR="${TVM_BASE_DIR}/build"
TVM_BUILD_CONFIG="${TVM_BUILD_DIR}/config.cmake"
DLPACK_BASE_DIR="${TVM_BASE_DIR}/3rdparty/dlpack"
DLPACK_BUILD_DIR="${DLPACK_BASE_DIR}/build"
DLPACKCPP_HEADER_DIR="${DLPACK_BASE_DIR}/contrib/dlpack"
DLPACKCPP_INSTALL_DIR="/usr/local/include/"

# install dependencies
apt-get update
apt-get install -y --no-install-recommends \
        ca-certificates ninja-build git python \
        cmake libopenblas-dev g++ llvm-8 llvm-8-dev clang-9 \
        python3.8 python3.8-dev python3-setuptools python3-pip antlr4 \
        build-essential

# install opencl
apt-get install -y --no-install-recommends \
    mesa-opencl-icd opencl-headers clinfo ocl-icd-opencl-dev
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf
echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# install vulkan (and opengl to have vulkan work nicely with nvidia)
apt-get install -y --no-install-recommends \
    libvulkan1 libvulkan-dev mesa-vulkan-drivers spirv-headers spirv-tools \
    libglvnd0 libgl1 libglx0 libegl1 libgles2
mkdir -p /usr/share/vulkan/icd.d
echo "{ \"file_format_version\" : \"1.0.0\", \"ICD\": { \"library_path\":
    \"libGLX_nvidia.so.0\" } }" > /usr/share/vulkan/icd.d/nvidia_icd.json
mkdir -p /usr/share/glvnd/egl_vendor.d
echo "{ \"file_format_version\" : \"1.0.0\", \"ICD\": { \"library_path\":
    \"libEGL_nvidia.so.0\" } }" > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# link clang-9 to be the default clang
update-alternatives --install /usr/bin/clang clang /usr/bin/clang-9 100

# install cross-compile toolchain
if [ "$(uname -i)" != "aarch64" ]; then
  apt-get install -y --no-install-recommends \
    g++-aarch64-linux-gnu \
    gcc-aarch64-linux-gnu
fi

rm -rf /var/lib/apt/lists/*
python3 -m pip install --upgrade pip
pip3 install mypy orderedset "antlr4-python3-runtime>=4.7,<4.8" \
  psutil "xgboost==1.5.*" tornado cython

# clone tvm and create build directory
git clone --branch ${TVM_VERSION} --recursive \
    https://github.com/apache/tvm ${TVM_BASE_DIR}
# Apply bugfix from https://github.com/apache/tvm/pull/13341
git -C ${TVM_BASE_DIR} cherry-pick a16a8904833e9c72aa7571ca336e781d89c128aa --no-commit
mkdir -p ${TVM_BUILD_DIR}

# copy a default configuration file
cp ${TVM_BASE_DIR}/cmake/config.cmake ${TVM_BUILD_DIR}

# turn on features
{
    echo "set(INSTALL_DEV TRUE)"
    echo "set(USE_LLVM llvm-config-8)"
    echo "set(USE_SORT ON)"
    echo "set(USE_GRAPH_RUNTIME ON)"
    echo "set(USE_BLAS openblas)"
    echo "set(USE_OPENCL ON)"
    echo "set(USE_VULKAN ON)"
} >> ${TVM_BUILD_CONFIG}
if [[ -d "/usr/local/cuda" ]]; then
    echo "set(USE_CUDA ON)" >> ${TVM_BUILD_CONFIG}
fi

# build and install tvm
pushd ${TVM_BUILD_DIR}
cmake ${TVM_BASE_DIR} -G Ninja
ninja
ninja install
popd

# install python extensions
pushd ${TVM_BASE_DIR}/python
python3 setup.py install
popd

# install dlpack headers
mkdir -p ${DLPACK_BUILD_DIR}
pushd ${DLPACK_BUILD_DIR}
cmake ${DLPACK_BASE_DIR} -G Ninja
ninja
ninja install
popd

# install dlpackcpp headers
cp -r ${DLPACKCPP_HEADER_DIR} ${DLPACKCPP_INSTALL_DIR}

# clean up
rm -rf ${TVM_BASE_DIR}

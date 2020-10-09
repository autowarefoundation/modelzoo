#! /usr/bin/env bash
# Copyright 2020 Autoware Foundation. All rights reserved.
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

TVM_VERSION_TAG="v0.6.1"
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
        python3.6 python3.6-dev python3-setuptools python3-pip antlr4

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
pip3 install mypy orderedset "antlr4-python3-runtime>=4.7,<4.8"

# clone tvm and create build directory
git clone --branch ${TVM_VERSION_TAG} --recursive \
    https://github.com/apache/incubator-tvm ${TVM_BASE_DIR}
mkdir -p ${TVM_BUILD_DIR}

# copy a default configuration file
cp ${TVM_BASE_DIR}/cmake/config.cmake ${TVM_BUILD_DIR}

# turn on features
echo "set(INSTALL_DEV TRUE)" >> ${TVM_BUILD_CONFIG}
echo "set(USE_LLVM llvm-config-8)" >> ${TVM_BUILD_CONFIG}
echo "set(USE_SORT ON)" >> ${TVM_BUILD_CONFIG}
echo "set(USE_GRAPH_RUNTIME ON)" >> ${TVM_BUILD_CONFIG}
echo "set(USE_BLAS openblas)" >> ${TVM_BUILD_CONFIG}
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

pushd ${TVM_BASE_DIR}/topi/python
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

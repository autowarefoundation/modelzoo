# Copyright 2021-2022 Arm Ltd., Apex.AI, Inc.
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

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project(execute_model)

find_package( OpenCV REQUIRED )

file(GLOB SRC_FILES "*.cpp")

set(TVM_ROOT /usr/tvm)
set(DMLC_CORE ${TVM_ROOT}/3rdparty/dmlc-core)
set(DLPACK ${TVM_ROOT}/3rdparty/dlpack)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")

add_executable(${CMAKE_PROJECT_NAME}
    ${ROS_NODE_FILE}
    ${SRC_FILES}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${OpenCV_LIBS}
    dl
    pthread
    tvm_runtime
)

target_include_directories (${CMAKE_PROJECT_NAME} PRIVATE
    ${TVM_ROOT}/include
    ${DMLC_CORE}/include
    ${DLPACK}/include
)

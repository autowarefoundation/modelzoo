#
# Copyright (c) 2020-2022, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

project( example_pipeline )

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

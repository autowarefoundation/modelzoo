#! /usr/bin/env bash
# Copyright 2020-2021 Autoware Foundation. All rights reserved.
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

IMAGE_NAME="autoware/model-zoo-tvm-cli"
TAG_NAME="local"
FROM_ARG="ubuntu:18.04"
TARGET_PLATFORM="arm64"
if [[ $(uname -a) == *"x86_64"* ]]; then
    TARGET_PLATFORM="amd64"
fi
CUDA_ENABLED="false"
CI_BUILD="false"

function usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "    -c,--cuda              Build TVM cli with cuda enabled."
    echo "    -h,--help              Display the usage and exit."
    echo "    -i,--image-name <name> Set docker images name."
    echo "                           Default: $IMAGE_NAME"
    echo "    -t,--tag <tag>         Tag use for the docker images."
    echo "                           Default: $TAG_NAME"
    echo "    --platform <platform>  Set target platform. Possible values: amd64, arm64."
    echo "                           Default: $TARGET_PLATFORM"
    echo "    --ci                   Enable CI-relevant build options"
    echo ""
}

OPTS=$(getopt --options chi:t: \
         --long cuda,help,image-name:,tag:,platform:,ci \
         --name "$0" -- "$@")
eval set -- "$OPTS"

while true; do
  case $1 in
    -c|--cuda)
      CUDA_ENABLED="true"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -i|--image-name)
      IMAGE_NAME="$2"
      shift 2
      ;;
    -t|--tag)
      TAG_NAME="$2"
      shift 2
      ;;
    --platform)
      TARGET_PLATFORM="$2"
      shift 2
      ;;
    --ci)
      CI_BUILD="true"
      shift 1
      ;;
    --)
      if [ -n "$2" ];
      then
        echo "Invalid parameter: $2"
        exit 1
      fi
      break
      ;;
    *)
      echo "Invalid option"
      exit 1
      ;;
  esac
done

if [ "$CUDA_ENABLED" == "true" ]; then
  FROM_ARG="nvidia/cuda-arm64:11.1-devel-ubuntu18.04"
  if [[ "${TARGET_PLATFORM}" == "amd64" ]]; then
    FROM_ARG="nvidia/cuda:10.1-devel-ubuntu18.04"
  fi
fi

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

DOCKER_FILE="Dockerfile.${TARGET_PLATFORM}"
BUILD_CONTEXT_DIR=${SCRIPT_PATH}

EXTRA_BUILD_ARGS=""
if [ "${CI_BUILD}" == "true" ]; then
  EXTRA_BUILD_ARGS="--cache-from=${IMAGE_NAME}:${TAG_NAME} --cache-to=type=inline,mode=max"
fi

# Build image with all dependencies and tvm_cli
export DOCKER_CLI_EXPERIMENTAL=enabled
docker buildx build -f "${SCRIPT_PATH}"/"${DOCKER_FILE}" \
            --build-arg FROM_ARG="${FROM_ARG}" \
            -t "${IMAGE_NAME}":"${TAG_NAME}" \
            --platform ${TARGET_PLATFORM} \
            --progress plain \
            --load \
            ${EXTRA_BUILD_ARGS} \
            "${BUILD_CONTEXT_DIR}"

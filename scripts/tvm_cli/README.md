# TVM Command Line Interface

A command line tool that compiles neural network models using
[TVM](https://github.com/apache/incubator-tvm).

## Usage

In all the subsequent commands, if a GPU needs to be enabled, the docker image
must be run with the correct flags, e.g. [--gpus 0] or [--gpus all] when running
an Nvidia GPU with proprietary drivers and
[--device /dev/dri --group-add video] otherwise.

```bash
$ docker run -it --rm -v `pwd`:`pwd` -w `pwd` \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware/model-zoo-tvm-cli:latest -h
```

### Compiling a model in the model zoo

```bash
$ export MODEL_DIR=<path to the folder containing definition.yaml>
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware/model-zoo-tvm-cli:latest \
        compile \
        --config ${MODEL_DIR}/definition.yaml \
        --output_path <output folder>
```

The output will consist of these file:

- `deploy_lib.so` contains compiled operators required by the network to be
  used with the TVM runtime
- `deploy_param.params` contains trained weights for the network to be used
  with the TVM runtime
- `deploy_graph.json` contains the compute graph defining relationship between
  the operators to be used with the TVM runtime
- `inference_engine_tvm_config.hpp` contains declaration of a structure with
  configuration for the TVM runtime C++ API.

The target device can be set with the `--target` parameter.

### Tuning a model in the model zoo

```bash
$ export MODEL_DIR=<path to the folder containing definition.yaml and \
                    AutoTVM_config.py>
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware/model-zoo-tvm-cli:latest \
        tune \
        --config ${MODEL_DIR}/definition.yaml \
        --output_path <output folder> \
        --autotvm_config ${MODEL_DIR}/AutoTVM_config.py
```

The output will consist of a .log file whose name can be specified in the
AutoTVM_config.py file. Examples of this file are provided in
`/modelzoo/AutoTVM_config_example_cuda.py` and
`/modelzoo/AutoTVM_config_example_llvm.py`

## Validation script

A testing script is provided. The script automatically detects all the .yaml
definition files in a user-specified path, executes the compilation of the
model corresponding to each file, and checks the output afterwards. The test
corresponding to a certain .yaml file is only executed if the 'enable_testing'
field is set to true.

The tests need to be run inside a container. The user is required to specify
the folder containing the .yaml files using the -v option. This folder will be
searched recursively for all the **definition.yaml** files.

```bash
$ docker run -it --rm \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    -v /abs_path_to/modelzoo/:/tmp \
    autoware/model-zoo-tvm-cli:latest \
    test
```

The output will contain information regarding which tests were successful and
which weren't.

## Building the docker image

Instead of pulling the docker image, it can be built locally using the
`build.sh` script:

```
Usage: ./scripts/tvm_cli/build.sh [OPTIONS]
    -c,--cuda              Build TVM cli with cuda enabled.
    -h,--help              Display the usage and exit.
    -i,--image-name <name> Set docker images name.
                           Default: autoware/model-zoo-tvm-cli
    -t,--tag <tag>         Tag use for the docker images.
                           Default: local
    --platform <platform>  Set target platform. Possible values: amd64, arm64.
                           Default: {native platform}
    --ci                   Enable CI-relevant build options
```

Here an example to build the image with default parameters:

```bash
$ # From root of the model zoo repo
$ ./scripts/tvm_cli/build.sh

...

Successfully built 547afbbfd193
Successfully tagged autoware/model-zoo-tvm-cli:local
```

The previous commands are then used with `:local` instead of `:latest`.

*Note:* If CUDA is needed, the `build.sh` script must be invoked with the `-c`
argument. In all the docker commands shown, if CUDA needs to be enabled, the
docker image must be run with a flag which exposes the gpu, e.g.
[--gpus 0] or [--gpus all].

*Note:* Cross-compilation is possible but experimental. Follow the
[buildx](https://github.com/docker/buildx#building-multi-platform-images)
instructions to setup your system.

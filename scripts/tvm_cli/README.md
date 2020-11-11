# TVM Command Line Interface

A command line tool that compiles neural network models using
[TVM](https://github.com/apache/incubator-tvm).

## Usage

In all the subsequent commands, if CUDA needs to be enabled, the docker image
must be run with a flag which exposes the gpu, e.g. [--gpus 0] or [--gpus all].

The CLI can now be invoked as a container

```bash
$ docker run -it --rm -v `pwd`:`pwd` -w `pwd` \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware/model-zoo-tvm-cli:latest -h
```

To compile a model in the model zoo

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

# Validation script

A testing script is provided. The script automatically detects all the .yaml
definition files in a user-specified path, executes the compilation of the
model corresponding to each file, and checks the output afterwards. The test
corresponding to a certain .yaml file is only executed if the 'enable_testing'
field is set to true.

## Usage

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

# Obtaining/Building the docker image

Pull a docker image that contains all the dependencies and scripts needed to
run the tool. Alternatively, the image can be built locally.

```bash
$ docker pull autoware/model-zoo-tvm-cli
```

Instead of pulling the docker image, it can be built locally.

```bash
$ # From root of the model zoo repo
$ docker build -f scripts/tvm_cli/Dockerfile \
               -t autoware/model-zoo-tvm-cli:local \
               scripts/tvm_cli
```

If CUDA is needed, the appropriate Dockerfile needs to be used instead.

```bash
$ # From root of the model zoo repo
$ docker build -f scripts/tvm_cli/Dockerfile.cuda \
               -t autoware/model-zoo-tvm-cli:local \
               scripts/tvm_cli
```

If the image is built locally, all the command in the *Usage* sections must be
run using `:local` instead of `:latest`.

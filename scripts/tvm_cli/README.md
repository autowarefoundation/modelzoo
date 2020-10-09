# TVM Command Line Interface

A command line tool that compiles neural network models using
[TVM](https://github.com/apache/incubator-tvm).

## Usage

Build a docker image that contains all the dependencies and scripts needed to
run the tool.

```bash
$ # From root of the model zoo repo
$ docker build -f scripts/tvm_cli/Dockerfile \
               -t autoware-model-zoo/tvm_cli:local \
               scripts/tvm_cli
```

The CLI can now be invoked as a container

```bash
$ docker run -it --rm -v `pwd`:`pwd` -w `pwd` \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware-model-zoo/tvm_cli:local -h
```

To compile a model in the model zoo

```bash
$ export MODEL_DIR=<path to the folder containing definition.yaml>
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware-model-zoo/tvm_cli:local \
        --config ${MODEL_DIR}/definition.yaml \
        --output_path <output folder>
```

The output will consist of these file:

- `deploy_lib.so` contains compiled operators required by the network to be used
  with the TVM runtime
- `deploy_param.params` contains trained weights for the network to be used with
  the TVM runtime
- `deploy_graph.json` contains the compute graph defining relationship between
  the operators to be used with the TVM runtime
- `inference_engine_tvm_config.hpp` contains declaration of a structure with
  configuration for the TVM runtime C++ API.

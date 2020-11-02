# TVM Command Line Interface

A command line tool that compiles neural network models using
[TVM](https://github.com/apache/incubator-tvm).

## Usage

Pull a docker image that contains all the dependencies and scripts needed to run
the tool. Alternatively, the image can be built locally, see the
[Building the docker image](#building-the-docker-image) paragraph.

```bash
$ docker pull autoware/model-zoo-tvm-cli
```

If CUDA is needed, the appropriate Dockerfile needs to be used instead:

```bash
$ # From root of the model zoo repo
$ docker build -f scripts/tvm_cli/Dockerfile.cuda \
               -t autoware-model-zoo/tvm_cli:local \
               scripts/tvm_cli
```

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

### Building the docker image

Instead of pulling the docker image, it can be built locally.

```bash
$ # From root of the model zoo repo
$ docker build -f scripts/tvm_cli/Dockerfile \
               -t autoware/model-zoo-tvm-cli:local \
               scripts/tvm_cli
```

The previous commands are then used with `:local` instead of `:latest`.
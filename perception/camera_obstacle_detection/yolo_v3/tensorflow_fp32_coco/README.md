# YOLO V3 Darknet Conversion to Keras

The network definition and weights come from [darknet
website](https://pjreddie.com/darknet/yolo/). It is converted to the keras model using [this](https://github.com/qqwweee/keras-yolo3) repository.

It has been converted to onnx format
using [tf2onnx](https://github.com/onnx/tensorflow-onnx).

## Executing the model 

All commands should be run from the root of the model zoo directory.

1.Compile a local image of `autoware/model-zoo-tvm-cli`

```bash
$./scripts/tvm_cli/build.sh
```

2.Compile the model by running the TVM CLI script in a docker container

```bash
$ MODEL_DIR=$(pwd)/perception/camera_obstacle_detection/yolo_v3/tensorflow_fp32_coco/
$ export MODEL_DIR
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware/model-zoo-tvm-cli:local \
        compile \
        --config ${MODEL_DIR}/definition.yaml \
        --output_path ${MODEL_DIR}/execute_model/build
```

3.Compile the c++ pipeline

```bash
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR}/execute_model/build \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --entrypoint "" \
    autoware/model-zoo-tvm-cli:local \
    bash -c "cmake .. && make -j"
```

4.Download a sample image and copy some files needed for decoding detections

```bash
$ curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg \
    > ${MODEL_DIR}/execute_model/build/test_image_0.jpg
$ cp ${MODEL_DIR}/model_files/labels.txt ${MODEL_DIR}/execute_model/build/
$ cp ${MODEL_DIR}/model_files/anchors.csv ${MODEL_DIR}/execute_model/build/
```

5.Run the detection pipeline inside a docker container. The output result can be obtained in two ways:

- **Save as an image**: saves the result of the pipeline as an image file in the build directory, the filename `output.jpg` can be changed in the command if needed:

    ```bash
    $ docker run \
        -it --rm \
        --net=host \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -w ${MODEL_DIR}/execute_model/build \
        --entrypoint "" \
        autoware/model-zoo-tvm-cli:local \
            ./execute_model output.jpg
    ```

- **Display in a X11 window**: X draw calls are forwarded to the host so the detection results can be displayed in a X11 window.

    ```bash
    $ docker run \
        -it --rm \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v ${HOME}/.Xauthority:${HOME}/.Xauthority:rw \
        -e XAUTHORITY=${HOME}/.Xauthority \
        -e DISPLAY=$DISPLAY \
        --net=host \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -w ${MODEL_DIR}/execute_model/build \
        --entrypoint "" \
        autoware/model-zoo-tvm-cli:local \
            ./execute_model
    ```

For more information about getting the TVM docker image, see the TVM CLI
[documentation](../../../../scripts/tvm_cli/README.md).
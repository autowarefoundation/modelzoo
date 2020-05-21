# YOLO V2 Tiny Darknet Convert Tensorflow

This model is converted into tensorflow format directly from the original
darknet weights from the
[darknet website](https://pjreddie.com/darknet/yolov2/).

## Provenance

The network definition and weights come from [darknet
website](https://pjreddie.com/darknet/yolov2/). It is converted to tensorflow
pb format using [darkflow](https://github.com/thtrieu/darkflow). This
[script](https://github.com/ARM-software/ML-examples/blob/master/autoware-vision-detector/scripts/get_yolo_tiny_v2.sh)
was used to produce the final model file.

## Compile and run the example pipeline

All commands should be run from the root of the model zoo directory. First,
build the TVM docker image by following instructions in the TVM CLI
[documentation](../../../../scripts/tvm_cli/README.md).

Compile the model by running the TVM CLI script in a docker container.

```bash
$ export MODEL_DIR=`pwd`/perception/camera_obstacle_detection/yolo_v2_tiny/tensorflow_fp32_coco/
$ mkdir ${MODEL_DIR}/example_pipeline/build
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    autoware-model-zoo/tvm_cli:local \
        --config ${MODEL_DIR}/definition.yaml \
        --output_path ${MODEL_DIR}/example_pipeline/build
```

Compile the example c++ inference pipeline.

```bash
$ docker run \
    -it --rm \
    -v ${MODEL_DIR}:${MODEL_DIR} -w ${MODEL_DIR}/example_pipeline/build \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --entrypoint "" \
    autoware-model-zoo/tvm_cli:local \
    bash -c "cmake .. && make -j"
```

Download an example image and copy some files needed for decoding detections.

```bash
$ curl https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg \
    > ${MODEL_DIR}/example_pipeline/build/test_image_0.jpg
$ cp ${MODEL_DIR}/model_files/labels.txt ${MODEL_DIR}/example_pipeline/build/
$ cp ${MODEL_DIR}/model_files/anchors.csv ${MODEL_DIR}/example_pipeline/build/
```

run the detection pipeline inside a docker container. X draw calls are forwarded
to the host so the detection results can be displayed in a X11 window.

```bash
$ docker run \
    -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v ${HOME}/.Xauthority:${HOME}/.Xauthority:rw \
    -e XAUTHORITY=${HOME}/.Xauthority \
    -e DISPLAY=$DISPLAY \
    --net=host \
    -v ${MODEL_DIR}:${MODEL_DIR} \
    -w ${MODEL_DIR}/example_pipeline/build \
    --entrypoint "" \
    autoware-model-zoo/tvm_cli:local \
        ./example_pipeline
```

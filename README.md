# Autoware Model Zoo

A collection of machine-learned models for use in autonomous driving
applications.

## License

All source code published in this repository by default are licensed under
Apache 2.0. If contributions need to be made under a difference license, the
LICENSE body need to be included in the sub-folder of the model. License for the
model files can be specified using the license fields in
[definition.yaml](#filling-out-the-definition-file).

## Cloning The Model Zoo

You need to [install Git LFS](https://git-lfs.github.com/) for large file
support. To clone the zoo:

```sh
$ GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/autowarefoundation/modelzoo.git
$ cd modelzoo
$ git lfs install
$ git lfs pull
```

## Contributing Models

### Folder Structure

The zoo is organized in terms of the following folder structure:

```sh
autoware-model-zoo/<task area>/<sub task area>/<model name>/<model variant name>
```

- **task area** The general autonomous driving area that the model is trying to
  tackle. include perception, prediction and planning. The list will expand as
  we develop further the autoware stack.
- **sub task area** The specific sub task that the model is trained to do. E.g.
  camera_traffic_light_detection.
- **model name** indicates the name of the model architecture, e.g. yolo_v3,
  ssd_mobilenet.
- **model variant** this should be an unique name to identify a variant of the
  model. Variants might differ in training data-set, quantization, input size,
  etc. You should choose a model variant name that describe best your model and
  avoid any potential conflict.

### Setup LFS Tracking

`git lfs` identify the files to track by their filename extension. To check what
extensions are already tracked in this zoo:

```sh
$ git lfs track
```

If you want to contribute something that is not on the list, add the filetype of
large files. For tflite, you would use:

```sh
$ git lfs track "*.tflite"
$ git add .gitattributes
$ git commit -m "tflite files are now tracked by LFS"
```

### Adding Models

This is an example of adding a FP32 object recognition model `yolo_v2_tiny.pb`
to the repository:

1. Fork the repository and clone your fork follow the steps
   [here](#cloning-the-model-zoo).
2. Create a new branch `git checkout -b contrib/<your_name>/<feature_name>`.
3. Create a folder to hold the model following the
   [folder structure](#folder-structure). For this example, this would be
   `perception/camera_obstacle_detection/yolo_v2_tiny/tensorflow_fp32_coco`.
4. Put the model files into this folder:
   `perception/camera_obstacle_detection/yolo_v2_tiny/tensorflow_fp32_coco/model_files`.
5. Fill out the `definition.yaml` file and put it at the root of the folder
   `perception/camera_obstacle_detection/yolo_v2_tiny/tensorflow_fp32_coco`. See
   guidelines [here](#filling-out-the-definition-file).
6. Stage and commit all new files
   `git add perception/camera_obstacle_detection/yolo_v2_tiny/**/*`
7. Push the changes `git push -u origin contrib/<your_name>/<feature_name>` and
   create a
   [merge request](https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html).

### Filling out the definition file

In the folder of each of the model, there should be a file named
`definition.yaml`. You can find a template of the file
[here](definition_template.yaml). The template contains guide on how to fill out
each field. Some important fields are discussed in the following paragraphs but
all attempts should be made to fill out all the fields in the template file.

#### Framework

Should be one of the following: `TensorFlow`, `Caffe`, `TensorFlow Lite`,
`ONNX`. If you would like to add any other format, follow
[this step](#setup-lfs-tracking) and update documentation before making a merge
request.

#### Provenance

The provenance field should be an URL to the "upstream" location. If more
information is required, they can be added in the [README](#additional-metadata)
file.

#### Tensor size fields

A shape is listed for each input/output node, and the intention is to capture
the input/output shape expected by the model. Entries having the number "-1"
here refer to components for which any size is accepted.

#### Other fields

Please make every attempt at filling out all the fields in the
[template](definition_template.yaml). Feel free to add any additional custom
fields after the template for better documentation and provenance.

### Additional Metadata

If you feel there are other information that would be relevant to other users,
please add a `README.md` to the root of the model's folder. You can share in the
document:

1. Link and description of any script for training
1. Verbose description of how the training data is obtained and pre-processed
1. Verbose discussion of bias and limitation in the training data
1. Guide on how to do transfer learning using this model
1. Share performance metrics on inference hardware
1. Share accuracy metrics on well known datasets.

Do commit any scripts or code you can share in the same folder as the models.

### Example pipeline

It is a good idea to include some code to show the complete inference pipeline.
The code should include pre-processing and post-processing. These details are
often difficult to convey in documentation and is hence best presented as code.
Some example input data should also be provided to accompany the example
pipeline. E.g. an image, a point cloud file, etc.

## Changing existing models

**Do not overwrite or remove** an existing model without a proper deprecation
process. Once someone has started analyzing, benchmarking or testing a model,
they want to be able to refer to the model, and know that that reference will
continue to refer to the same model. Creating a new folder and deprecating the
old one is preferred over modifying existing models.

## TVM CLI

[TVM](https://github.com/apache/incubator-tvm) is a machine learning compiler
framework that compiles neutral network models to be executed on different
hardware back-ends. TVM enables models specified in a wide range of ML
frameworks to be run on a wide range of hardware devices.

In this model zoo, a CLI tool has been provided to compile models using TVM.
This forms part of the validation workflow of the zoo as well as model
deployment workflow for Autoware.

See [TVM CLI documentation](scripts/tvm_cli/README.md) on how to use the tool.

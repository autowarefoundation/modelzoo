# Point pillars PFE network

The pfe.onnx model file is a network trained using the Kitti 3D Object detection
evaluation 2017 dataset, it is a first stage network during the point pillars
computation algorithm.

## Provenance

The model file can be retrieved from this repository:
<https://github.com/k0suke-murakami/kitti_pretrained_point_pillars.git>

## Compile the model using the TVM cli

Place the pfe.onnx file in the model_files directory and run the TVM CLI
[documentation](../../../../scripts/tvm_cli/README.md) using the definition file
supplied in this folder.

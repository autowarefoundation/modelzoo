#! /usr/bin/env python3
#
# Copyright (c) 2020, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys
import onnx
import yaml
import tensorflow as tf
import tvm.relay.testing.tf as tf_testing
import tvm.relay as relay
from jinja2 import Environment, FileSystemLoader
from tvm.contrib import cc
from os import path

OUTPUT_NETWORK_MODULE_FILENAME = "deploy_lib.so"
OUTPUT_NETWORK_GRAPH_FILENAME = "deploy_graph.json"
OUTPUT_NETWORK_PARAM_FILENAME = "deploy_param.params"
OUTPUT_CONFIG_FILENAME = "inference_engine_tvm_config.hpp"

# This function checks the command-line arguments
# and reads the necessary info from the .yaml file.
def preprocess(args):
    # 'info' is the output dictionary
    info = {}

    info['model_path'] = args.model
    info['lanes'] = args.lanes
    info['device_type'] = args.device_type
    info['device_id'] = args.device_id
    info['cross_compile'] = args.cross_compile

    # .yaml file processing
    with open(args.config, 'r') as yml_file:
        yaml_dict = yaml.safe_load(yml_file)
        # Get path of model file
        if not info['model_path']:
            info['model_path'] = yaml_dict['network']['filename']
        if not info['model_path'].startswith('/'):
            yaml_file_dir = path.dirname(yml_file.name)
            info['model_path'] = path.join(yaml_file_dir, info['model_path'])
        # Get list of input names and shapes from .yaml file
        info['input_list'] = yaml_dict['network_parameters']['input_nodes']
        info['input_dict'] = {}                     # Used to compile the model
        for input_elem in info['input_list']:
            info['input_dict'][str(input_elem['name'])] = input_elem['shape']
        # Get input data type
        input_data_type = yaml_dict['network_parameters']['datatype']
        if input_data_type == 'float32':
            info['dtype_code'] = 'kDLFloat'
            info['dtype_bits'] = 32
        elif input_data_type == 'int8':
            info['dtype_code'] = 'kDLInt'
            info['dtype_bits'] = 8
        else:
            raise Exception('Specified input data type not supported')
        # Get list of output names and shapes from .yaml file
        info['output_list'] = yaml_dict['network_parameters']['output_nodes']
        info['output_names'] = []                   # Used to compile the model
        for output_elem in info['output_list']:
            info['output_names'].append(str(output_elem['name']))

    # Define the root directory and check if the specified output_path exists.
    # If not the corresponding directories are created.
    # Note: if output_path has not been specified by the user,
    # default to the 'filename' field from the .yaml file
    if args.output_path:
        info['output_path'] = args.output_path
    if not path.isdir(info['output_path']):
        os.makedirs(info['output_path'])

    # starting from the config file directory, take 4 levels of parent directory
    # as the namespace in the case of the model zoo these 4 levels correspond to
    # <task area>/<autonomous driving task>/<model name>/<model variant name>.
    model_dir = path.abspath(path.dirname(args.config))
    namespaces = model_dir.split(path.sep)
    if len(namespaces) < 4:
        info['namespace'] = model_dir
    else:
        info['namespace'] = path.sep.join(namespaces[-4:])

    return info

# This function generates the config .hpp file.
def generate_config_file(info):
    # Setup jinja template and write the config file
    root = path.dirname(path.abspath(__file__))
    templates_dir = path.join(root, 'templates')
    env = Environment( loader = FileSystemLoader(templates_dir),
                       keep_trailing_newline = True )
    template = env.get_template(OUTPUT_CONFIG_FILENAME + ".jinja2")
    filename = path.join(info['output_path'], OUTPUT_CONFIG_FILENAME)

    print('Writing pipeline configuration to', filename)
    with open(filename, 'w') as fh:
        fh.write(template.render(
            namespace = info['namespace'],
            network_module_path = path.join('.', OUTPUT_NETWORK_MODULE_FILENAME),
            network_graph_path = path.join('.', OUTPUT_NETWORK_GRAPH_FILENAME),
            network_params_path = path.join('.', OUTPUT_NETWORK_PARAM_FILENAME),
            tvm_dtype_code = info['dtype_code'],
            tvm_dtype_bits = info['dtype_bits'],
            tvm_dtype_lanes = info['lanes'],
            tvm_device_type = info['device_type'],
            tvm_device_id = info['device_id'],
            input_list = info['input_list'],
            output_list = info['output_list']
        ))

# This functions compiles the model.
def compile(info):
    if info['model_path'].endswith('.onnx'):
        is_onnx = True
    elif info['model_path'].endswith('.pb'):
        is_onnx = False
    else:
        raise Exception('Model file format not supported')

    # Load model
    if is_onnx:
        onnx_model = onnx.load(info['model_path'])
        mod, params = relay.frontend.from_onnx(onnx_model, info['input_dict'])
        optimization_level = 3
    else:
        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(info['model_path'], 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
                graph_def = sess.graph.as_graph_def()
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        input_shape_dict = {'DecodeJpeg/contents': info['input_list']}
        mod, params = relay.frontend.from_tensorflow(graph_def,
                                                shape=input_shape_dict,
                                                outputs=info['output_names'])
        optimization_level = 2

    # Set compilation params
    target = 'llvm'
    if info['cross_compile']:
        target += ' -target=aarch64-linux-gnu'

    # Compile model
    # Note opt_level cannot be higher than 2 because of a bug:
    # https://discuss.tvm.ai/t/tvm-0-6-1-compile-yolo-v2-tiny-fail-worked-in-v0-5-2/7244
    with relay.build_config(opt_level=optimization_level):
        graph, lib, params = relay.build(mod,
                                         target=target,
                                         params=params)

    # Write the compiled model to files
    output_model_path = path.join(info['output_path'],
                                  OUTPUT_NETWORK_MODULE_FILENAME)
    output_graph_path = path.join(info['output_path'],
                                  OUTPUT_NETWORK_GRAPH_FILENAME)
    output_param_path = path.join(info['output_path'],
                                  OUTPUT_NETWORK_PARAM_FILENAME)

    print('Writing library to', output_model_path)
    if info['cross_compile']:
        lib.export_library(output_model_path,
                           cc.build_create_shared_func(
                               options=['--target=aarch64-linux-gnu',
                                        '-march=armv8-a',
                                        '-mfpu=NEON'],
                               compile_cmd='/usr/bin/clang'))
    else:
        lib.export_library(output_model_path)

    print('Writing graph to', output_graph_path)
    with open(output_graph_path, 'w') as graph_file:
        graph_file.write(graph)

    print('Writing weights to', output_param_path)
    with open(output_param_path, 'wb') as param_file:
        param_file.write(relay.save_param_dict(params))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Compile model and configuration file (TVM)')
    requiredNamed = parser.add_argument_group('required arguments')
    requiredNamed.add_argument('--config',
                               help='Path to .yaml config file (input)',
                               required=True)
    requiredNamed.add_argument('--output_path',
                               help='Path where network module, '
                                    'network graph and network parameters '
                                    'will be stored',
                               required=True)
    parser.add_argument('--model',
                        help='Path to .onnx/.pb model file (input)')
    parser.add_argument('--device_type',
                        help='User-specified device type',
                        choices=['kDLCPU', 'kDLGPU', 'kDLCPUPinned',
                                 'kDLOpenCL', 'kDLVulkan', 'kDLMetal',
                                 'kDLVPI', 'kDLROCM', 'kDLExtDev'],
                        default='kDLCPU')
    parser.add_argument('--device_id',
                        help='User-specified device ID',
                        type=int,
                        default=1)
    parser.add_argument('--lanes',
                        help='Number of lanes, default value is 1',
                        type=int,
                        default=1)
    parser.add_argument('--cross_compile',
                        help='Set to cross compile for ArmV8a with NEON',
                        action='store_true',
                        default=False)

    # The dictionary 'info' contains all the information provided by the user
    # and the information found in the .yaml file.
    info = preprocess(parser.parse_args())
    compile(info)
    generate_config_file(info)

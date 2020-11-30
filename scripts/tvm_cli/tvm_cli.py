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
import tvm.relay.testing.tf as tf_testing
import tvm.relay as relay
from jinja2 import Environment, FileSystemLoader
from tvm.contrib import cc
from tvm import autotvm
import tvm
from os import path
import pytest
import tensorflow as tf
import numpy as np
import tvm.contrib.graph_runtime as runtime
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

OUTPUT_NETWORK_MODULE_FILENAME = "deploy_lib.so"
OUTPUT_NETWORK_GRAPH_FILENAME = "deploy_graph.json"
OUTPUT_NETWORK_PARAM_FILENAME = "deploy_param.params"
OUTPUT_CONFIG_FILENAME = "inference_engine_tvm_config.hpp"

# Utility function: definition.yaml file processing
def yaml_processing(config, info):
    with open(config, 'r') as yml_file:
        yaml_dict = yaml.safe_load(yml_file)
        # Get path of model file
        info['model'] = yaml_dict['network']['filename']
        if not info['model'].startswith('/'):
            yaml_file_dir = path.dirname(yml_file.name)
            info['model'] = path.join(yaml_file_dir, info['model'])
        # Get list of input names and shapes from .yaml file
        info['input_list'] = yaml_dict['network_parameters']['input_nodes']
        info['input_dict'] = {}                     # Used to compile the model
        for input_elem in info['input_list']:
            info['input_dict'][str(input_elem['name'])] = input_elem['shape']
        # Get input data type
        info['input_data_type'] = yaml_dict['network_parameters']['datatype']
        if info['input_data_type'] == 'float32':
            info['dtype_code'] = 'kDLFloat'
            info['dtype_bits'] = 32
        elif info['input_data_type'] == 'int8':
            info['dtype_code'] = 'kDLInt'
            info['dtype_bits'] = 8
        else:
            raise Exception('Specified input data type not supported')
        # Get list of output names and shapes from .yaml file
        info['output_list'] = yaml_dict['network_parameters']['output_nodes']
        info['output_names'] = []                   # Used to compile the model
        for output_elem in info['output_list']:
            info['output_names'].append(str(output_elem['name']))
    return info

# Utility function to load the model
def get_network(info):
    if info['model'].endswith('.onnx'):
        onnx_model = onnx.load(info['model'])
        mod, params = relay.frontend.from_onnx(onnx_model, info['input_dict'])
    elif info['model'].endswith('.pb'):
        with tf.compat.v1.Session() as sess:
            with tf.io.gfile.GFile(info['model'], 'rb') as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                input_map = {}
                for index, (name, shape) in enumerate(
                                                info['input_dict'].items()):
                    tf_new_image = tf.compat.v1.placeholder(
                        shape=[1 if x == -1 else x for x in shape],
                        dtype=info['input_data_type'],
                        name=name)
                    input_map["input:"+str(index)] = tf_new_image
                tf.import_graph_def(graph_def,
                                    name='',
                                    input_map = input_map)
                graph_def = sess.graph.as_graph_def()
                graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        input_shape_dict = {'DecodeJpeg/contents': info['input_list']}
        mod, params = relay.frontend.from_tensorflow(
            graph_def,
            shape=input_shape_dict,
            outputs=info['output_names'])
    else:
        raise Exception('Model file format not supported')

    # Transform data layout to what is expected by CUDA hardware, i.e. NCHW
    if info['target'] == 'cuda':
        desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
        seq = tvm.transform.Sequential(
            [relay.transform.RemoveUnusedFunctions(),
            relay.transform.ConvertLayout(desired_layouts)])
        with tvm.transform.PassContext(opt_level=3):
            mod = seq(mod)

    return mod, params

# This function checks the command-line arguments and reads the necessary info
# from the .yaml file, it's used when the compile option is selected
def compilation_preprocess(args):
    # 'info' is the output dictionary
    info = {}

    info['lanes'] = args.lanes
    info['device_type'] = args.device_type
    info['device_id'] = args.device_id
    info['target'] = args.target
    info['cross_compile'] = args.cross_compile
    info['autotvm_log'] = args.autotvm_log

    info = yaml_processing(args.config, info)

    # Define the root directory and check if the specified output_path exists.
    # If not the corresponding directories are created.
    # Note: if output_path has not been specified by the user,
    # default to the 'filename' field from the .yaml file
    if args.output_path:
        info['output_path'] = args.output_path
    if not path.isdir(info['output_path']):
        os.makedirs(info['output_path'])

    # Starting from the config file directory, take 4 levels of parent
    # directory as the namespace in the case of the model zoo these 4 levels
    # correspond to <task area>/<autonomous driving task>/<model name>/<model
    # variant name>.
    model_dir = path.abspath(path.dirname(args.config))
    namespaces = model_dir.split(path.sep)
    if len(namespaces) < 4:
        info['namespace'] = model_dir
    else:
        info['namespace'] = path.sep.join(namespaces[-4:])

    return info

# This functions compiles the model
def compile_model(info):
    mod, params = get_network(info)

    # Set compilation params
    if info['cross_compile']:
        if info['target'] == 'cuda':
            raise Exception('cuda cross-compilation not supported yet')
        info['target'] += ' -target=aarch64-linux-gnu'

    # Compile model
    with autotvm.apply_history_best(info['autotvm_log']):
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod,
                                             target=info['target'],
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
            network_module_path = path.join('.',
                                            OUTPUT_NETWORK_MODULE_FILENAME),
            network_graph_path = path.join('.',
                                           OUTPUT_NETWORK_GRAPH_FILENAME),
            network_params_path = path.join('.',
                                            OUTPUT_NETWORK_PARAM_FILENAME),
            tvm_dtype_code = info['dtype_code'],
            tvm_dtype_bits = info['dtype_bits'],
            tvm_dtype_lanes = info['lanes'],
            tvm_device_type = info['device_type'],
            tvm_device_id = info['device_id'],
            input_list = info['input_list'],
            output_list = info['output_list']
        ))

# This function checks the command-line arguments and reads the necessary info
# from the .yaml file, it's used when the tune option is selected
def tuning_preprocess(args):
    # 'info' is the output dictionary
    info = {}

    info['tuner'] = args.tuner
    info['n_trial'] = args.n_trial
    info['early_stopping'] = args.early_stopping
    info['evaluate_inference_time'] = args.evaluate_inference_time

    info = yaml_processing(args.config, info)

    # Define the root directory and check if the specified output_path exists.
    # If not the corresponding directories are created.
    # Note: if output_path has not been specified by the user,
    # default to the 'filename' field from the .yaml file
    if args.output_path:
        info['output_path'] = args.output_path
    if not path.isdir(info['output_path']):
        os.makedirs(info['output_path'])

    # Import the AutoTVM config file
    sys.path.append(os.path.dirname(os.path.abspath(args.autotvm_config)))
    autotvm_config_file = os.path.basename(args.autotvm_config)
    import importlib
    info['cfg'] = importlib.import_module(autotvm_config_file[:-3])

    return info

# This function performs the tuning of a model
def tune_model(info):
    def tune_cuda(mod, params):
        def tune_tasks(
            tasks,
            target,
            measure_option,
            tuner,
            n_trial,
            early_stopping,
            log_filename,
            use_transfer_learning
        ):
            # Overwrite AutoTVM_config contents if the user provides the
            # corresponding arguments
            if info['tuner'] is not None:
                tuner = info['tuner']
            if info['n_trial'] is not None:
                n_trial = info['n_trial']
            if info['early_stopping'] is not None:
                early_stopping = info['early_stopping']

            # create tmp log file
            tmp_log_file = log_filename + ".tmp"
            if os.path.exists(tmp_log_file):
                os.remove(tmp_log_file)

            for i, tsk in enumerate(reversed(tasks)):
                prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

                # create tuner
                if tuner == "xgb" or tuner == "xgb-rank":
                    tuner_obj = XGBTuner(tsk, loss_type="rank")
                elif tuner == "ga":
                    tuner_obj = GATuner(tsk, pop_size=100)
                elif tuner == "random":
                    tuner_obj = RandomTuner(tsk)
                elif tuner == "gridsearch":
                    tuner_obj = GridSearchTuner(tsk)
                else:
                    raise ValueError("Invalid tuner: " + tuner)

                if use_transfer_learning and os.path.isfile(tmp_log_file):
                    tuner_obj.load_history(
                        autotvm.record.load_from_file(tmp_log_file))

                # do tuning
                tsk_trial = min(n_trial, len(tsk.config_space))
                tuner_obj.tune(
                    n_trial=tsk_trial,
                    early_stopping=early_stopping,
                    measure_option=measure_option,
                    callbacks=[
                        autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                        autotvm.callback.log_to_file(tmp_log_file),
                    ],
                )

            # pick best records to a cache file
            autotvm.record.pick_best(tmp_log_file,
                                     path.join(info['output_path'],
                                               log_filename))
            os.remove(tmp_log_file)

        # extract workloads from relay program
        print("Extract tasks...")
        tasks = autotvm.task.extract_from_program(
            mod["main"],
            target=info['target'],
            params=params,
            ops=(relay.op.get("nn.conv2d"),))

        # run tuning tasks
        print("Tuning...")
        tune_tasks(tasks, **tuning_opt)

        print("The .log file has been saved in " +
              path.join(info['output_path'], tuning_opt['log_filename']))

        if info['evaluate_inference_time']:
            # compile kernels with history best records
            with autotvm.apply_history_best(
                path.join(info['output_path'],
                          tuning_opt['log_filename'])):
                print("Compile...")
                with tvm.transform.PassContext(opt_level=3):
                    lib = relay.build_module.build(mod,
                                                   target=info['target'],
                                                   params=params)

                # load parameters
                ctx = tvm.context(info['target'], 0)
                module = runtime.GraphModule(lib["default"](ctx))
                for name, shape in info['input_dict'].items():
                    data_tvm = tvm.nd.array(
                        (np.random.uniform(
                            size=[1 if x == -1 else x for x in shape]))
                        .astype(info['input_data_type']))
                    module.set_input(name, data_tvm)

                # evaluate
                print("Evaluate inference time cost...")
                ftimer = module.module.time_evaluator("run",
                                                      ctx,
                                                      number=1,
                                                      repeat=600)
                prof_res = np.array(ftimer().results) * 1000  # convert to ms
                print("Mean inference time (std dev): %.2f ms (%.2f ms)"
                      % (np.mean(prof_res), np.std(prof_res)))

    tuning_opt = info['cfg'].tuning_options
    info['target'] = tuning_opt['target']
    mod, params = get_network(info)
    if info['target'] == 'cuda':
        tune_cuda(mod, params)
    else:
        raise Exception('Tuning target not supported yet')

if __name__ == '__main__':
    import argparse

    def compile():
        parser = argparse.ArgumentParser(
            description='Compile a model using TVM',
            usage='''tvm_cli compile [<args>]''')
        requiredNamed = parser.add_argument_group('required arguments')
        requiredNamed.add_argument('--config',
                                   help='Path to .yaml config file (input)',
                                   required=True)
        requiredNamed.add_argument('--output_path',
                                   help='Path where network module, '
                                        'network graph and network parameters '
                                        'will be stored',
                                   required=True)
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
        parser.add_argument('--target',
                            help='Set the compilation target',
                            choices=['llvm', 'cuda'],
                            default='llvm')
        parser.add_argument('--cross_compile',
                            help='Set to cross compile for ArmV8a with NEON',
                            action='store_true',
                            default=False)
        parser.add_argument('--autotvm_log',
                            help='Path to an autotvm .log file, can speed up '
                                 'inference')

        parsed_args = parser.parse_args(sys.argv[2:])

        # The dictionary 'info' contains all the information provided by the
        # user and the information found in the .yaml file
        info = compilation_preprocess(parsed_args)
        compile_model(info)
        generate_config_file(info)

    def tune():
        parser = argparse.ArgumentParser(
            description='Tune a model using AutoTVM, at the moment only CUDA '
                        'target is supported',
            usage='''tvm_cli tune [<args>]''')
        requiredNamed = parser.add_argument_group('required arguments')
        requiredNamed.add_argument('--config',
                                   help='Path to .yaml config file (input)',
                                   required=True)
        requiredNamed.add_argument('--output_path',
                                   help='Path where the output .log file will '
                                        'be stored',
                                   required=True)
        requiredNamed.add_argument('--autotvm_config',
                                   help='Path to an autotvm config file, see '
                                        'AutoTVM_config_example.py',
                                   required=True)
        parser.add_argument('--tuner',
                            help='Specify the tuner to be used, overrides '
                                 '--autotvm_config contents',
                            choices=['xgb', 'xgb-rank', 'ga', 'random',
                                     'gridsearch'])
        parser.add_argument('--n_trial',
                            help='Maximum number of configurations to try, '
                                 'overrides --autotvm_config contents',
                            type=int)
        parser.add_argument('--early_stopping',
                            help='Early stop the tuning when not finding '
                                 'better configs in this number of trials, '
                                 'overrides --autotvm_config contents',
                            type=int)
        parser.add_argument('--evaluate_inference_time',
                            help='Set to perform an inference time evaluation '
                                 'after the tuning phase',
                            action='store_true',
                            default=False)

        parsed_args = parser.parse_args(sys.argv[2:])

        # The dictionary 'info' contains all the information provided by the
        # user and the information found in the .yaml file
        info = tuning_preprocess(parsed_args)
        tune_model(info)

    def test():
        parser = argparse.ArgumentParser(
            description='Launch the validation script',
            usage='''tvm_cli test [-h]''')
        parser.parse_args(sys.argv[2:])
        pytest.main(['-v'])

    parser = argparse.ArgumentParser(
        description='Compile model and configuration file (TVM)',
        usage='''<command> [<args>]
Commands:
    compile    Compile a model using TVM
    tune       Tune a model using AutoTVM
    test       Launch the validation script''')
    parser.add_argument('command', help='Subcommand to run')
    parsed_args = parser.parse_args(sys.argv[1:2])
    if parsed_args.command not in locals():
        print('Unrecognized command')
        parser.print_help()
        exit(1)

    # Invoke method with same name as the argument passed
    locals()[parsed_args.command]()

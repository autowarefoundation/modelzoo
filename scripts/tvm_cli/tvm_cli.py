#! /usr/bin/env python3
#
# Copyright (c) 2020-2022, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import sys
from os import path
import importlib
import subprocess
import onnx
import yaml
import tvm
# import tvm.relay.testing.tf as tf_testing
import tvm.contrib.graph_runtime as runtime
from tvm import relay
from tvm import autotvm
from tvm.contrib import cc
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
from jinja2 import Environment, FileSystemLoader
import pytest
# import tensorflow as tf
import numpy as np

OUTPUT_PREPROCESSING_MODULE_FILENAME = "preprocess.so"
OUTPUT_NETWORK_MODULE_FILENAME = "deploy_lib.so"
OUTPUT_NETWORK_GRAPH_FILENAME = "deploy_graph.json"
OUTPUT_NETWORK_PARAM_FILENAME = "deploy_param.params"
OUTPUT_CONFIG_FILENAME = "inference_engine_tvm_config.hpp"
OUTPUT_PREPROCESSING_CONFIG_FILENAME = "preprocessing_inference_engine_tvm_config.hpp"

TARGETS_DEVICES = {
    'llvm':'kDLCPU',
    'cuda':'kDLCUDA',
    'opencl':'kDLOpenCL',
    'vulkan':'kDLVulkan',
}
GPU_TARGETS = ['cuda', 'opencl', 'vulkan']

def yaml_helper(info, yaml_dict, process, type_str):
    if type_str == 'input':
        list_name = 'input_list'
        dict_name = 'input_dict'
        node_name = 'input_nodes'
    else:
        list_name = 'output_list'
        dict_name = 'output_dict'
        node_name = 'output_nodes'
    if process == '':
        info[list_name] = yaml_dict['network_parameters'][node_name]
        list_elem = yaml_dict['network_parameters'][node_name]
        info[dict_name] = {}
        pre_input_dict = info[dict_name]
        pre_input_list = info[list_name]
    else:
        info[process][list_name] = yaml_dict[process][node_name]
        list_elem = yaml_dict[process][node_name]
        info[process][dict_name] = {}
        pre_input_dict = info[process][dict_name]
        pre_input_list = info[process][list_name]
    for idx in range(len(list_elem)):
        input_elem = list_elem[idx]
        input_name = str(input_elem['name'])
        pre_input_dict[input_name] = {}
        pre_input_dict[input_name] = input_elem['shape']
        pre_input_list[idx]['lanes'] = 1
        if input_elem['datatype'] == 'float32':
            pre_input_list[idx]['dtype_code'] = 'kDLFloat'
            pre_input_list[idx]['dtype_bits'] = 32
        elif input_elem['datatype'] == 'int8':
            pre_input_list[idx]['dtype_code'] = 'kDLInt'
            pre_input_list[idx]['dtype_bits'] = 8
        elif input_elem['datatype'] == 'int32':
            pre_input_list[idx]['dtype_code'] = 'kDLInt'
            pre_input_list[idx]['dtype_bits'] = 32
        else:
            raise Exception('Specified input data type not supported')

def yaml_processing(config, info):
    '''Utility function: definition.yaml file processing'''
    with open(config, 'r', encoding='utf-8') as yml_file:
        yaml_dict = yaml.safe_load(yml_file)
        # Get path of model file
        info['model'] = yaml_dict['network']['filename']
        if not info['model'].startswith('/'):
            yaml_file_dir = path.dirname(yml_file.name)
            info['model'] = path.join(yaml_file_dir, info['model'])
        yaml_helper(info, yaml_dict, '', 'input')
        # # Get list of input names and shapes from .yaml file
        # info['input_list'] = yaml_dict['network_parameters']['input_nodes']
        # info['input_dict'] = {}                     # Used to compile the model
        # for input_elem in info['input_list']:
        #     info['input_dict'][str(input_elem['name'])] = input_elem['shape']
        # # Get input data type
        # info['input_data_type'] = yaml_dict['network_parameters']['datatype']
        # if info['input_data_type'] == 'float32':
        #     info['dtype_code'] = 'kDLFloat'
        #     info['dtype_bits'] = 32
        # elif info['input_data_type'] == 'int8':
        #     info['dtype_code'] = 'kDLInt'
        #     info['dtype_bits'] = 8
        # else:
        #     raise Exception('Specified input data type not supported')
        # Get list of output names and shapes from .yaml file
        yaml_helper(info, yaml_dict, '', 'output')
        # info['output_list'] = yaml_dict['network_parameters']['output_nodes']
        # info['output_names'] = []                   # Used to compile the model
        # for output_elem in info['output_list']:
        #     info['output_names'].append(str(output_elem['name']))
        if 'preprocessing' in yaml_dict:
            info['preprocessing'] = {}
            info['preprocessing']['network_name'] = yaml_dict['preprocessing']['module_name']
            yaml_file_dir = path.dirname(yml_file.name)
            info['preprocessing']['module'] = path.join(yaml_file_dir, yaml_dict['preprocessing']['module'])
            yaml_helper(info, yaml_dict, 'preprocessing', 'input')
            yaml_helper(info, yaml_dict, 'preprocessing', 'output')

    return info

def get_network(info):
    '''Utility function to load the model'''
    if info['model'].endswith('.onnx'):
        onnx_model = onnx.load(info['model'])
        mod, params = relay.frontend.from_onnx(onnx_model, info['input_dict'])
    # elif info['model'].endswith('.pb'):
    #     with tf.compat.v1.Session() as sess:
    #         with tf.io.gfile.GFile(info['model'], 'rb') as f:
    #             graph_def = tf.compat.v1.GraphDef()
    #             graph_def.ParseFromString(f.read())
    #             input_map = {}
    #             for index, (name, shape) in enumerate(
    #                                             info['input_dict'].items()):
    #                 tf_new_image = tf.compat.v1.placeholder(
    #                     shape=[1 if x == -1 else x for x in shape],
    #                     dtype=info['input_data_type'],
    #                     name=name)
    #                 input_map["input:"+str(index)] = tf_new_image
    #             tf.import_graph_def(graph_def,
    #                                 name='',
    #                                 input_map = input_map)
    #             graph_def = sess.graph.as_graph_def()
    #             graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    #     input_shape_dict = {'DecodeJpeg/contents': info['input_list']}
    #     mod, params = relay.frontend.from_tensorflow(
    #         graph_def,
    #         shape=input_shape_dict,
    #         outputs=info['output_dict'].keys())
    else:
        raise Exception('Model file format not supported')

    # Transform data layout to what is expected by CUDA hardware, i.e. NCHW.
    # The same code is used in the llvm case too, as this allows for a simpler
    # handling of AutoTVM tuning. For tuning on x86, the NCHWc layout would be
    # the best choice, but TVM doesn't fully support it yet
    if info['target'] in GPU_TARGETS:
        desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
    elif info['target'].startswith('llvm'):
        desired_layouts = {'nn.conv2d': ['NCHW', 'default']}
    else:
        raise Exception('Target not supported')
    seq = tvm.transform.Sequential(
        [relay.transform.RemoveUnusedFunctions(),
         relay.transform.ConvertLayout(desired_layouts)])
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    return mod, params

def compilation_preprocess(args):
    '''
    This function checks the command-line arguments and reads the necessary info
    from the .yaml file, it's used when the compile option is selected
    '''
    # 'info' is the output dictionary
    info = {}

    info['lanes'] = args.lanes
    info['device_type'] = TARGETS_DEVICES[args.target]
    info['device_id'] = args.device_id
    info['target'] = args.target
    info['cross_compile'] = args.cross_compile
    info['autotvm_log'] = args.autotvm_log
    info['header_extension'] = '.h' if args.autoware_version == 'ai' else '.hpp'

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
        info['network_name'] = model_dir
    else:
        info['namespace'] = path.sep.join(namespaces[-4:])
        info['network_name'] = namespaces[-2]

    # Get version information from the most recent tag.
    # Default to 0.0.0 when the git tag information can't be accessed.
    info['modelzoo_version'] = [0, 0, 0]
    modelzoo_rootdir = path.abspath(path.dirname(__file__) + "/../../")
    if path.isdir(path.join(modelzoo_rootdir, '.git')):
        cmd = 'git describe --tags --match [0-9]*'.split()
        try:
            version = subprocess.check_output(cmd).decode().strip().split('-', maxsplit=1)[0]
            version = version.split('.')
            if len(version) == 3:
                info['modelzoo_version'] = version
            else:
                print(
                    f"Unexpected git tag, descriptor version defaults to "
                    f"{info['modelzoo_version']}"
                )
        except subprocess.CalledProcessError:
            print(
                f"No git tag information, descriptor version defaults to "
                f"{info['modelzoo_version']}"
            )

    return info

def compile_model(info):
    '''This functions compiles the model'''
    mod, params = get_network(info)

    # Set compilation params
    if info['cross_compile']:
        if info['target'] in GPU_TARGETS:
            raise Exception(info['target'] + ' cross-compilation not supported yet')
        info['target'] += ' -mtriple=aarch64-linux-gnu'

    # Compile model
    if info['autotvm_log'] is not None:
        if info['target'].startswith('llvm'):
            cm = autotvm.apply_graph_best(info['autotvm_log'])
        elif info['target'] in GPU_TARGETS:
            cm = autotvm.apply_history_best(info['autotvm_log'])
        with cm:
            with relay.build_config(opt_level=3):
                graph, lib, params = relay.build(mod,
                                                 target=info['target'],
                                                 params=params)
    else:
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
                           cc.cross_compiler(
                               compile_func='/usr/bin/clang',
                               options=['--target=aarch64-linux-gnu',
                                        '-march=armv8-a',
                                        '-mfpu=NEON']))
    else:
        lib.export_library(output_model_path)

    print('Writing graph to', output_graph_path)
    with open(output_graph_path, 'w', encoding='utf-8') as graph_file:
        graph_file.write(graph)

    print('Writing weights to', output_param_path)
    with open(output_param_path, 'wb') as param_file:
        param_file.write(relay.save_param_dict(params))

def compile_preprocessing_lib(info):
    if 'preprocessing' in info:
        spec = importlib.util.spec_from_file_location("preprocessing", info['preprocessing']['module'])   
        module = importlib.util.module_from_spec(spec)       
        spec.loader.exec_module(module)

        rt_lib = tvm.build(module.PreprocessingModule, target=info['target'])
        output_preprocessing_path = path.join(info['output_path'],
                                    OUTPUT_PREPROCESSING_MODULE_FILENAME)
        rt_lib.export_library(output_preprocessing_path)

def generate_config_file(info):
    '''This function generates the config .hpp file'''
    # Setup jinja template and write the config file
    root = path.dirname(path.abspath(__file__))
    templates_dir = path.join(root, 'templates')
    env = Environment( loader = FileSystemLoader(templates_dir),
                       keep_trailing_newline = True )
    template = env.get_template(OUTPUT_CONFIG_FILENAME + ".jinja2")
    filename = path.join(info['output_path'], OUTPUT_CONFIG_FILENAME)

    print('Writing pipeline configuration to', filename)
    with open(filename, 'w', encoding='utf-8') as fh:
        fh.write(template.render(
            namespace = info['namespace'],
            header_extension = info['header_extension'],
            modelzoo_version = info['modelzoo_version'],
            network_name = info['network_name'],
            network_backend = info['target'],
            network_module_path = path.join('.',
                                            OUTPUT_NETWORK_MODULE_FILENAME),
            network_graph_path = path.join('.',
                                        OUTPUT_NETWORK_GRAPH_FILENAME),
            network_params_path = path.join('.',
                                            OUTPUT_NETWORK_PARAM_FILENAME),
            tvm_device_type = info['device_type'],
            tvm_device_id = info['device_id'],
            input_list = info['input_list'],
            output_list = info['output_list']
        ))
    
    if 'preprocessing' in info:
        filename = path.join(info['output_path'], OUTPUT_PREPROCESSING_CONFIG_FILENAME)
        print('Writing pipeline configuration to', filename)
        with open(filename, 'w', encoding='utf-8') as fh:
            fh.write(template.render(
                namespace = info['namespace'],
                header_extension = info['header_extension'],
                modelzoo_version = info['modelzoo_version'],
                network_name = info['network_name'] + '_' + info['preprocessing']['network_name'],
                network_backend = info['target'],
                network_module_path = path.join('.',
                                                OUTPUT_PREPROCESSING_MODULE_FILENAME),
                network_graph_path = './',
                network_params_path = './',
                tvm_device_type = info['device_type'],
                tvm_device_id = info['device_id'],
                input_list = info['input_list'],
                output_list = info['output_list']
            ))

def tuning_preprocess(args):
    '''
    This function checks the command-line arguments and reads the necessary info
    from the .yaml file, it's used when the tune option is selected
    '''
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
    info['cfg'] = importlib.import_module(autotvm_config_file[:-3])

    return info

def tune_model(info):
    '''This function performs the tuning of a model'''
    def tune_kernels(
        tasks,
        tuning_opt
    ):
        tuner = tuning_opt['tuner']
        n_trial = tuning_opt['n_trial']
        early_stopping = tuning_opt['early_stopping']

        # Overwrite AutoTVM_config contents if the user provides the
        # corresponding arguments
        if info['tuner'] is not None:
            tuner = info['tuner']
        if info['n_trial'] is not None:
            n_trial = info['n_trial']
        if info['early_stopping'] is not None:
            early_stopping = info['early_stopping']

        for i, tsk in enumerate(reversed(tasks)):
            prefix = f"[Task {i + 1:2d}/{len(tasks):2d}] "

            # create tuner
            if tuner in ('xgb', 'xgb-rank'):
                tuner_obj = XGBTuner(tsk, loss_type="rank")
            elif tuner == "ga":
                tuner_obj = GATuner(tsk, pop_size=100)
            elif tuner == "random":
                tuner_obj = RandomTuner(tsk)
            elif tuner == "gridsearch":
                tuner_obj = GridSearchTuner(tsk)
            else:
                raise ValueError("Invalid tuner: " + tuner)

            # do tuning
            tsk_trial = min(n_trial, len(tsk.config_space))
            tuner_obj.tune(
                n_trial=tsk_trial,
                early_stopping=early_stopping,
                measure_option=tuning_opt['measure_option'],
                callbacks=[
                    autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                    autotvm.callback.log_to_file(path.join(
                        info['output_path'],
                        tuning_opt['log_filename'])),
                ],
            )

    # Use graph tuner to achieve graph level optimal schedules
    # Set use_DP=False if it takes too long to finish.
    def tune_graph(graph,
                   records,
                   opt_sch_file,
                   min_exec_graph_tuner,
                   use_DP=True):
        target_op = [
            relay.op.get('nn.conv2d'),
        ]
        Tuner = DPTuner if use_DP else PBQPTuner
        executor = Tuner(graph,
                         {name:[1 if x == -1 else x for x in shape]
                          for (name,shape) in info['input_dict'].items()},
                         records,
                         target_op,
                         info['target'])
        executor.benchmark_layout_transform(min_exec_num=min_exec_graph_tuner)
        executor.run()
        executor.write_opt_sch2record_file(path.join(info['output_path'],
                                                     opt_sch_file))

    tuning_opt = info['cfg'].tuning_options
    info['target'] = tuning_opt['target']
    mod, params = get_network(info)

    # extract workloads from relay program
    print("Extract tasks...")
    tasks = autotvm.task.extract_from_program(
        mod["main"],
        target=info['target'],
        params=params,
        ops=(relay.op.get("nn.conv2d"),))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, tuning_opt)
    if info['target'].startswith('llvm'):
        opt_sch_file = tuning_opt['log_filename'][:-4] + '_graph_opt.log'
        tune_graph(
            mod['main'],
            path.join(info['output_path'], tuning_opt['log_filename']),
            path.join(info['output_path'], opt_sch_file),
            tuning_opt['min_exec_graph_tuner'])

    if info['target'] in GPU_TARGETS:
        print("The .log file has been saved in " +
              path.join(info['output_path'], tuning_opt['log_filename']))
    elif info['target'].startswith('llvm'):
        print("The .log file has been saved in " +
              path.join(info['output_path'], opt_sch_file))

    if info['evaluate_inference_time']:
        if info['target'] in GPU_TARGETS:
            cm = autotvm.apply_history_best(
                path.join(info['output_path'],
                          tuning_opt['log_filename']))
        elif info['target'].startswith('llvm'):
            cm = autotvm.apply_graph_best(
                path.join(info['output_path'], opt_sch_file))
        # compile
        with cm:
            print("Compile...")
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build_module.build(mod,
                                               target=info['target'],
                                               params=params)

            # load parameters
            if info['target'] in GPU_TARGETS:
                ctx = tvm.context(info['target'], info['device_id'])
            elif info['target'].startswith('llvm'):
                ctx = tvm.cpu()
            module = runtime.GraphModule(lib["default"](ctx))
            for name, value in info['input_dict'].items():
                shape = value['shape']
                data_tvm = tvm.nd.array(
                    (np.random.uniform(
                        size=[1 if x == -1 else x for x in shape]))
                    .astype(info['input_data_type']))
                module.set_input(name, data_tvm)

            # evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run",
                                                  ctx,
                                                  number=10,
                                                  repeat=60)
            prof_res = np.array(ftimer().results) * 1000
            print(f"Mean inference time (std dev): "
                  f"{np.mean(prof_res):.2f} ms ({np.std(prof_res):.2f} ms)")

if __name__ == '__main__':
    import argparse

    def compile():
        '''Compiles a model using TVM'''
        parser = argparse.ArgumentParser(
            description='Compile a model using TVM',
            usage='''tvm_cli compile [<args>]''',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        requiredNamed = parser.add_argument_group('required arguments')
        requiredNamed.add_argument('--config',
                                   help='Path to .yaml config file (input)',
                                   required=True)
        requiredNamed.add_argument('--output_path',
                                   help='Path where network module, '
                                        'network graph and network parameters '
                                        'will be stored',
                                   required=True)
        targets = list(TARGETS_DEVICES)
        parser.add_argument('--target',
                            help='Compilation target',
                            choices=targets,
                            default=targets[0])
        parser.add_argument('--device_id',
                            help='Device ID',
                            type=int,
                            default=0)
        parser.add_argument('--lanes',
                            help='Number of lanes',
                            type=int,
                            default=1)
        parser.add_argument('--cross_compile',
                            help='Cross compile for ArmV8a with NEON',
                            action='store_true',
                            default=False)
        parser.add_argument('--autotvm_log',
                            help='Path to an autotvm .log file, can speed up '
                                 'inference')
        parser.add_argument('--autoware_version',
                            help='Targeted Autoware version',
                            choices=['ai', 'auto'],
                            default='auto')

        parsed_args = parser.parse_args(sys.argv[2:])

        # The dictionary 'info' contains all the information provided by the user
        # and the information found in the .yaml file
        try:
            info = compilation_preprocess(parsed_args)
            compile_model(info)
            compile_preprocessing_lib(info)
            generate_config_file(info)
        except Exception as e:
            print('Exception: '+ str(e))
            return 1

        return 0

    def tune():
        '''Tunes a model using AutoTVM'''
        parser = argparse.ArgumentParser(
            description='Tune a model using AutoTVM',
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
                                 'overrides --autotvm_config contents.',
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
        '''Launches the validation script'''
        parser = argparse.ArgumentParser(
            description='Launch the validation script',
            usage='''tvm_cli test [-h]''')
        parser.parse_args(sys.argv[2:])
        return pytest.main(['-v'])

    main_parser = argparse.ArgumentParser(
        description='Compile model and configuration file (TVM)',
        usage='''<command> [<args>]
Commands:
    compile    Compile a model using TVM
    tune       Tune a model using AutoTVM
    test       Launch the validation script''')
    main_parser.add_argument('command', help='Subcommand to run')
    main_parsed_args = main_parser.parse_args(sys.argv[1:2])
    if main_parsed_args.command not in locals():
        print('Unrecognized command')
        main_parser.print_help()
        exit(1)

    # Invoke method with same name as the argument passed
    exit(locals()[main_parsed_args.command]())

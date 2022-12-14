import os
import sys
from os import path
import importlib
import subprocess
# import onnx
import yaml
import tvm
import importlib.util

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
        pre_input_dict[input_name]['shape'] = input_elem['shape']
        pre_input_dict[input_name]['input_data_type'] = input_elem['datatype']
        pre_input_dict[input_name]['lanes'] = 1
        if pre_input_dict[input_name]['input_data_type'] == 'float32':
            pre_input_dict[input_name]['dtype_code'] = 'kDLFloat'
            pre_input_dict[input_name]['dtype_bits'] = 32
        elif pre_input_dict[input_name]['input_data_type'] == 'int8':
            pre_input_dict[input_name]['dtype_code'] = 'kDLInt'
            pre_input_dict[input_name]['dtype_bits'] = 8
        elif pre_input_dict[input_name]['input_data_type'] == 'int32':
            pre_input_dict[input_name]['dtype_code'] = 'kDLInt'
            pre_input_dict[input_name]['dtype_bits'] = 32
        else:
            raise Exception('Specified input data type not supported')
        pre_input_list[idx]['dtype_code'] = pre_input_dict[input_name]['dtype_code']
        pre_input_list[idx]['dtype_bits'] = pre_input_dict[input_name]['dtype_bits']
        pre_input_list[idx]['lanes'] = pre_input_dict[input_name]['lanes']

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

info = yaml_processing('/home/xinyuwang/adehome/modelzoo/perception/lidar_obstacle_detection/centerpoint_backbone/onnx_centerpoint_backbone/definition.yaml',{})
# print(info['preprocessing']['module'])
print(info['input_dict'])
print("\n")
print(info['input_list'])
# print(info['output_list'])
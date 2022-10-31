#! /usr/bin/env python3
#
# Copyright (c) 2020-2022, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import shutil
import subprocess
from glob import glob
from os import path
import tempfile
from pathlib import Path
import yaml
import pytest

MOUNT_PATH = path.abspath(path.dirname(__file__) + '/../../..')
BACKENDS = ['llvm', 'vulkan']

def run_tvm_cli(config_path, output_folder, extra_run_args):
    '''Execute tvm_cli and check the return code'''
    run_arg = [path.join(MOUNT_PATH, "./scripts/tvm_cli/tvm_cli.py"),
               'compile',
               '--config', config_path,
               '--output_path', output_folder]
    run_arg += extra_run_args
    proc = subprocess.run(run_arg, check=True)
    assert proc.returncode == 0

    # Check if the files have been generated
    assert os.path.exists(os.path.join(output_folder, 'deploy_graph.json'))
    assert os.path.exists(os.path.join(output_folder, 'deploy_lib.so'))
    assert os.path.exists(os.path.join(output_folder, 'deploy_param.params'))
    assert os.path.exists(os.path.join(output_folder,
                                       'inference_engine_tvm_config.hpp'))

# Create a list containing the paths to all the .yaml files found in the
# mounted folder
definition_files = glob(MOUNT_PATH + '/**/definition.yaml', recursive=True)

# Create a dictionary containing the network names associated to their path, for
# the files which have enable_testing: true
networks_to_compile = {}
for definition_file in definition_files:
    with open(definition_file, 'r', encoding='utf-8') as yaml_file:
        yaml_dict = yaml.safe_load(yaml_file)
        if yaml_dict['enable_testing']:
            name = definition_file.split(path.sep)[-3]
            networks_to_compile[name] = definition_file

root_folder = path.join(MOUNT_PATH, 'neural_networks')

# Parameterizing the test_tvm_cli function, we generate separate tests for
# every .yaml file: one for each target backend.
@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('network_name', list(networks_to_compile))
def test_tvm_cli(backend, network_name):
    '''Executes a test for each backend-network combination'''
    # Create a directory for every model
    dir_name = '-'.join([network_name, os.uname().machine, backend])
    output_folder = path.join(root_folder, dir_name)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    config_path = networks_to_compile[network_name]
    extra_run_args = ['--target', backend]
    run_tvm_cli(config_path, output_folder, extra_run_args)

# Parameterizing the test_tvm_cli function, we generate separate tests for
# every .yaml file.
@pytest.mark.skipif(os.uname().machine == 'aarch64',
                    reason='would cross-compile to itself')
@pytest.mark.parametrize('network_name', list(networks_to_compile))
def test_tvm_cli_cross_compile(network_name):
    '''Executes a cross compilation test for each network'''
    # Create a directory for every model
    output_folder = tempfile.mkdtemp()

    config_path = networks_to_compile[network_name]
    extra_run_args = ['--cross_compile']
    run_tvm_cli(config_path, output_folder, extra_run_args)

    # Delete the output folder
    shutil.rmtree(output_folder)

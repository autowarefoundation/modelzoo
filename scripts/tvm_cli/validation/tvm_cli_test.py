#! /usr/bin/env python3
#
# Copyright (c) 2020, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#

import os
import shutil
import subprocess
import yaml
import pytest
from glob import glob
from os import path
import tempfile

MOUNT_PATH = '/tmp'

# Create a list containing the paths to all the .yaml files found in the
# mounted folder
definition_files = glob(MOUNT_PATH + '/**/definition.yaml', recursive=True)

# Create a list containing only those files which have enable_testing: true
files_to_test = []
for definition_file in definition_files:
    with open(definition_file, 'r') as yaml_file:
        if yaml.safe_load(yaml_file)['enable_testing']:
            files_to_test.append(definition_file)

# Generate a separate test for every .yaml file
@pytest.mark.parametrize('definition_file', files_to_test)
def test_tvm_cli(definition_file):
    # Create a temporary directory for every model
    output_folder = tempfile.mkdtemp()

    # Execute tvm_cli and check the return code
    proc = subprocess.run(
        [path.join(MOUNT_PATH, "./scripts/tvm_cli/tvm_cli.py"),
         'compile',
         '--config', definition_file,
         '--output_path', output_folder])
    assert proc.returncode == 0

    # Check if the files have been generated
    assert os.path.exists(os.path.join(output_folder, 'deploy_graph.json'))
    assert os.path.exists(os.path.join(output_folder, 'deploy_lib.so'))
    assert os.path.exists(os.path.join(output_folder, 'deploy_param.params'))
    assert os.path.exists(os.path.join(output_folder,
                                       'inference_engine_tvm_config.hpp'))

    # Delete the output folder
    shutil.rmtree(output_folder)

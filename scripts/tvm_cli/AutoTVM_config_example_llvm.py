#! /usr/bin/env python3
#
# Copyright (c) 2020, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
import os
from tvm import autotvm

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 8
os.environ['TVM_NUM_THREADS'] = str(num_threads)

tuning_options = {
    # Replace 'llvm' with the correct target of your CPU.
    # For example, for AWS EC2 c5 instance with Intel Xeon
    # Platinum 8000 series, the target should be 'llvm -mcpu=skylake'.
    # For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
    # 'llvm -mcpu=core-avx2'.
    'target': 'llvm',
    'log_filename': 'my_log_file.log',
    'tuner': 'xgb',
    'n_trial': 1000,
    'early_stopping': 600,
    # min_exec_graph_tuner is specific to cpu tuning
    'min_exec_graph_tuner': 1,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(
            number=1, repeat=10, min_repeat_ms=0, enable_cpu_cache_flush=True
        ),
    ),
}

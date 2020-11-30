#! /usr/bin/env python3
#
# Copyright (c) 2020, Arm Limited and Contributors. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
from tvm import autotvm

tuning_options = {
    'target': 'cuda',
    'log_filename': 'my_log_file.log',
    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,
    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20,
                                   repeat=3,
                                   timeout=4,
                                   min_repeat_ms=150),
    ),
    'use_transfer_learning': False
}

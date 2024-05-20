#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging as log
import sys
import argparse
import openvino as ov


def param_to_string(parameters) -> str:
    """Convert a list / tuple of parameters returned from IE to a string."""
    if isinstance(parameters, (list, tuple)):
        return ', '.join([str(x) for x in parameters])
    else:
        return str(parameters)


def device():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

    # --------------------------- Step 1. Initialize OpenVINO Runtime Core --------------------------------------------
    core = ov.Core()

    # --------------------------- Step 2. Get metrics of available devices --------------------------------------------
    log.info('Available devices:')
    for device in core.available_devices:
        log.info(f'{device} :')
        log.info('\tSUPPORTED_PROPERTIES:')
        for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
            if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
                try:
                    property_val = core.get_property(device, property_key)
                except TypeError:
                    property_val = 'UNSUPPORTED TYPE'
                log.info(f'\t\t{property_key}: {param_to_string(property_val)}')
        log.info('')

    # -----------------------------------------------------------------------------------------------------------------
    return 0

def datatype():
    core = ov.Core()
    cpu_optimization_capabilities = core.get_property("CPU", ov.properties.device.capabilities())
    log.info(f'\t\tcpu_optimization_capabilities: {cpu_optimization_capabilities}')

def inference_precision_hint(model):
    core = ov.Core()
    # from optimum.intel.openvino import OVModelForCausalLM
    # from optimum.intel import OVModelForCausalLM
    # model = OVModelForCausalLM.from_pretrained(model, device="CPU")
    compiled_model = core.compile_model(model, "CPU")
    inference_precision = core.get_property("CPU", ov.properties.hint.inference_precision())
    log.info(f'\t\inference_precision: {inference_precision}')

if __name__ == '__main__':
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h',
                        '--help',
                        action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-t',
                        '--type',
                        required=True,
                        type=str)
    args = parser.parse_args()
    if args.type == 'datatype':
        sys.exit(datatype())
    elif args.type == 'precision':
        inference_precision_hint(model="/home/sdp/models/llama2-7b-ov/openvino_model.xml")
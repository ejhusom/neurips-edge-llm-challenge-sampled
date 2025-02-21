#!/usr/bin/env python3
# -*- coding: utf-8 -*-
def get_dataset_name(file_name):
    parts = file_name.split('_')
    return parts[0]

def get_model_name(file_name):
    parts = file_name.split(".")[0].split('_')
    return "_".join(parts[1:])

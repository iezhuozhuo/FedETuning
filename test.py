"""Test some code snippets"""

import os
import sys
import itertools as it
from loguru import logger
from multiprocessing import Pool
from configs.tuning import hyperparameter_grid


def run_process(proc):
    os.system(proc)

run_dirs = sys.argv[1]
fl_algorithm = sys.argv[2]
task_name = sys.argv[3]
tuning_type = sys.argv[4]
port_start = int(sys.argv[5])
device = sys.argv[6]

device_idx_list = [idx for idx in device.split(",")]
n_gpu = len(device_idx_list)
world_size = 3
logger.info(f"world_size is {world_size}")

if task_name == "conll":
    max_seq = 32
    data_file = "fedner"
    dataset_name = "ner"
    metric_name = "conll"
    model_output_mode = "token_classification"
else:
    max_seq = 128
    data_file = "fedglue"
    dataset_name = "glue"
    metric_name = "glue"
    model_output_mode = "seq_classification"

logger.info(f"{task_name}'s max_seq is {max_seq}")

cmds = []
gpu_index = 0
hyper_parameter = hyperparameter_grid[tuning_type]
for parameter in it.product(*list(hyper_parameter.values())):
    specific_parameter_dict = {key: parameter[list(hyper_parameter.keys()).index(key)]
                               for key in list(hyper_parameter.keys())}
    if "lora_rank" in specific_parameter_dict:
        specific_parameter_dict["lora_alpha"] = specific_parameter_dict["lora_rank"]
    # print(specific_parameter_dict)

    options = []
    for key, value in specific_parameter_dict.items():
        options.extend(["--" + key, str(value)])
    print(" ".join(options))

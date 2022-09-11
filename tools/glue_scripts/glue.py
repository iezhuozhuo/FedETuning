import os
import pickle
import argparse
from loguru import logger
from sklearn.model_selection import train_test_split
from transformers import glue_output_modes as output_modes

from utils import make_sure_dirs
from tools.glue_scripts.partition import GlueDataPartition
from tools.glue_scripts.glue_utils import glue_processors as processors


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--task", default=None, type=str, required=True,
        help="Task name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory to save partition or raw data")
    parser.add_argument("--clients_num", default=None, type=int, required=True,
        help="All clients numbers")
    parser.add_argument("--alpha", default=None, type=float,
        help="The label skew degree.")
    parser.add_argument("--overwrite", default=None, type=int,
        help="overwrite")

    args = parser.parse_args()
    return args


def load_glue_examples(args):
    task_name = args.task.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()

    # if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
    # HACK(label indices are swapped in RoBERTa pretrained model)
    # label_list[1], label_list[2] = label_list[2], label_list[1]

    train_examples = processor.get_train_examples(args.data_dir)
    valid_examples = processor.get_dev_examples(args.data_dir)
    test_examples = processor.get_test_examples(args.data_dir)

    return train_examples, valid_examples, test_examples, output_mode, label_list


def get_partition_data(examples, num_classes, num_clients, label_vocab, dir_alpha, partition, ):
    targets = [example.label for example in examples]
    clients_partition_data = GlueDataPartition(
        targets=targets, num_classes=num_classes, num_clients=num_clients,
        label_vocab=label_vocab, dir_alpha=dir_alpha, partition=partition, verbose=False
    )
    # assert (len(clients_partition_data) == num_clients,
    #         "The partition function is wrong, please check")

    partition_data = {}
    for idx in range(len(clients_partition_data)):
        client_idxs = clients_partition_data[idx]
        partition_data[idx] = client_idxs
    return partition_data


def convert_glue_to_federated_pkl(args):

    logger.info("reading examples ...")

    if os.path.isfile(args.output_data_file):
        logger.info(f"Examples in {args.output_data_file} have existed ...")
        with open(args.output_data_file, "rb") as file:
            data = pickle.load(file)
        train_examples, valid_examples, test_examples = data["train"], data["valid"], data["test"]
        output_mode, label_list = data["output_mode"], data["label_list"]
    else:
        logger.info(f"Generating examples from {args.data_dir} ...")
        train_examples, original_valid_examples, original_test_examples, output_mode, label_list \
            = load_glue_examples(args)
        # we need to split original valid_examples into new valid and test sets
        original_valid_examples_idx = [i for i in range(len(original_valid_examples))]
        original_valid_examples_label = [example.label for example in original_valid_examples]
        valid_idx, test_idx, valid_y, test_y = train_test_split(
            original_valid_examples_idx, original_valid_examples_label,
            test_size=0.5, random_state=42, stratify=original_valid_examples_label
        )
        valid_examples = [original_valid_examples[idx] for idx in valid_idx]
        test_examples = [original_valid_examples[idx] for idx in test_idx]

        data = {
            "train": train_examples, "valid": valid_examples, "test": test_examples,
            "output_mode": output_mode, "label_list": label_list
        }
        with open(args.output_data_file, "wb") as file:
            pickle.dump(data, file)

    logger.info("partition data ...")
    if os.path.isfile(args.output_partition_file):
        logger.info("loading partition data ...")
        with open(args.output_partition_file, "rb") as file:
            partition_data = pickle.load(file)
    else:
        partition_data = {}

    logger.info(f"partition data's keys: {partition_data.keys()}")

    if f"clients={args.clients_num}_alpha={args.alpha}" in partition_data and not args.overwrite:
        logger.info(f"Partition method 'clients={args.clients_num}_alpha={args.alpha}' has existed "
                    f"and overwrite={args.overwrite}, then skip")
    else:
        lable_mapping = {label: idx for idx, label in enumerate(label_list)}
        attribute = {"lable_mapping": lable_mapping, "label_list": label_list,
                     "clients_num": args.clients_num, "alpha": args.alpha,
                     "output_mode": output_mode
                     }
        clients_partition_data = {"train": get_partition_data(
            examples=train_examples, num_classes=len(label_list), num_clients=args.clients_num,
            label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
        ), "valid": get_partition_data(
            examples=valid_examples, num_classes=len(label_list), num_clients=args.clients_num,
            label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
        ), "test": get_partition_data(
            examples=test_examples, num_classes=len(label_list), num_clients=args.clients_num,
            label_vocab=label_list, dir_alpha=args.alpha, partition="dirichlet"
        ), "attribute": attribute}

        logger.info(f"writing clients={args.clients_num}_alpha={args.alpha} ...")
        partition_data[f"clients={args.clients_num}_alpha={args.alpha}"] = clients_partition_data

        with open(args.output_partition_file, "wb") as file:
            pickle.dump(partition_data, file)

    logger.info("end")


if __name__ == "__main__":
    logger.info("start...")
    args = parser_args()
    data_dir = args.data_dir
    output_dir = args.output_dir

    tasks = ["MRPC", "SST-2", "QNLI", "QQP", "MNLI", "CoLA", "RTE"]
    # tasks = ["MRPC"]
    client_nums = [100, 10]
    args.overwrite = True
    for task in tasks:
        for client_num in client_nums:
            args.clients_num = client_num
            args.task = task
            args.data_dir = os.path.join(data_dir, args.task)
            args.output_dir = os.path.join(output_dir, "fedglue")
            make_sure_dirs(args.output_dir)
            args.output_data_file = os.path.join(args.output_dir, f"{args.task.lower()}_data.pkl")
            args.output_partition_file = os.path.join(args.output_dir, f"{args.task.lower()}_partition.pkl")

            logger.info(f"clients_num: {args.clients_num}")
            logger.info(f"data_dir: {args.data_dir}")
            logger.info(f"output_dir: {args.output_dir}")
            logger.info(f"output_data_file: {args.output_data_file}")
            logger.info(f"output_partition_file: {args.output_partition_file}")

            convert_glue_to_federated_pkl(args)

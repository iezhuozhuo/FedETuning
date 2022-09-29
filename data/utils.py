"""Data Process Utils"""
from datasets import ClassLabel, load_dataset


def conll_examples_to_features(examples, tokenizer):
    ...
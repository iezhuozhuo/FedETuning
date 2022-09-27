"""Data Process Utils"""
from datasets import ClassLabel, load_dataset

raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )


# In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
# unique labels.
def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list

# If the labels are of type ClassLabel, they are already integers and we have the map stored somewhere.
# Otherwise, we have to get the list of labels manually.
labels_are_int = isinstance(features[label_column_name].feature, ClassLabel)
if labels_are_int:
    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
else:
    label_list = get_label_list(raw_datasets["train"][label_column_name])
    label_to_id = {l: i for i, l in enumerate(label_list)}

num_labels = len(label_list)


# Map that sends B-Xxx label to its I-Xxx counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

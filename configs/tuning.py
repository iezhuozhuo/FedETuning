"""Test some code snippets"""


def get_delta_config(delta_name):
    all_delta_config = {
        "adapter_roberta-base":
            {
                "delta_type": "adapter",
                "learning_rate": 3e-4,
                "unfrozen_modules": [
                    "deltas",
                    "layer_norm",
                    "final_layer_norm",
                    "classifier",
                ],
                "bottleneck_dim": 24,
            },
        'soft_prompt_roberta-base':
            {
                "delta_type": "soft_prompt",
                "learning_rate": 3e-2,
                "soft_token_num": 100,
                "unfrozen_modules": [
                    "deltas",
                    "classifier",
                ],
            },
        "lora_roberta-base":
            {
                "rte":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0005,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 80,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 32,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "qqp":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0005,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 25,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 16,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "mrpc":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0004,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 30,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 16,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas",
                            "layer_norm"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "mnli":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0005,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 30,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 16,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "cola":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0004,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 80,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 32,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "qnli":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0004,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 25,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 32,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    },
                "sst-2":
                    {
                        "delta_type": "lora",
                        "learning_rate": 0.0005,
                        "lora_alpha": 8,
                        "lora_rank": 8,
                        "non_linearity": "gelu_new",
                        "num_train_epochs": 60,
                        "per_device_eval_batch_size": 100,
                        "per_device_train_batch_size": 16,
                        "unfrozen_modules": [
                            "classifier",
                            "deltas"
                        ],
                        "warmup_ratio": 0.06,
                        "weight_decay": 0.1,
                    }

            },
        "bitfit_robert-base":
            {
                "delta_type": "bitfit",
                "learning_rate": 3e-4,
                "output_dir": "outputs/bitfit/roberta-base/",
                "unfrozen_modules": [
                    "classifier",
                    "deltas"
                ],
            }
    }
    return all_delta_config[delta_name]

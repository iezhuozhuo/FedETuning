"""Test some code snippets"""
import os
import torch
from loguru import logger
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader

from utils import pickle_read

# class CoLAPromptTemple(object):
#     def __init__(self, data_path):
#         self.data_path = data_path



# dataset = [  # For simplicity, there's only two examples
#     # text_a is the input text of the data, some other datasets may have multiple input sentences in one example.
#     InputExample(
#         guid=0,
#         text_a="Albert Einstein was one of the greatest intellects of his time.",
#     ),
#     InputExample(
#         guid=1,
#         text_a="The film was badly made.",
#     ),
# ]

class MyManualTemplate(ManualTemplate):
    def wrap_one_example(self,
                         example: InputExample):
        if self.text is None:
            raise ValueError("template text has not been initialized")

        text = self.incorporate_text_example(example)

        # not_empty_keys = example.keys()
        not_empty_keys = [key for key in example.__dict__.keys() if getattr(example, key) is not None]
        for placeholder_token in self.placeholder_mapping:
            not_empty_keys.remove(self.placeholder_mapping[placeholder_token]) # placeholder has been processed, remove
        if "meta" in not_empty_keys:
            not_empty_keys.remove('meta') # meta has been processed

        keys, values= ['text'], [text]
        for inputflag_name in self.registered_inputflag_names:
            keys.append(inputflag_name)
            v = None
            if hasattr(self, inputflag_name) and getattr(self, inputflag_name) is not None:
                v = getattr(self, inputflag_name)
            elif hasattr(self, "get_default_"+inputflag_name):
                v = getattr(self, "get_default_"+inputflag_name)()
                setattr(self, inputflag_name, v) # cache
            else:
                raise ValueError("""
                Template's inputflag '{}' is registered but not initialize.
                Try using template.{} = [...] to initialize
                or create an method get_default_{}(self) in your template.
                """.format(inputflag_name, inputflag_name, inputflag_name))

            if len(v) != len(text):
                raise ValueError("Template: len({})={} doesn't match len(text)={}."\
                    .format(inputflag_name, len(v), len(text)))
            values.append(v)
        wrapped_parts_to_tokenize = []
        for piece in list(zip(*values)):
            wrapped_parts_to_tokenize.append(dict(zip(keys, piece)))

        wrapped_parts_not_tokenize = {key: getattr(example, key) for key in not_empty_keys}
        return [wrapped_parts_to_tokenize, wrapped_parts_not_tokenize]

task_name = "cola"
raw_dataset_path = os.path.join("/workspace/data/fedglue/", f"{task_name}_data.pkl")
raw_data = pickle_read(raw_dataset_path)


plm, tokenizer, model_config, WrapperClass = load_plm("roberta", "/workspace/pretrain/nlp/roberta-base/")

promptTemplate = MyManualTemplate(
    # text='{"placeholder":"text_a"} It was {"mask"}',
    text='{"placeholder":"text_a"} This is {"mask"} .',
    tokenizer=tokenizer,
)

classes = [  # There are two classes in Sentiment Analysis, one for negative and one for positive
    "0",
    "1"
]
label_words = {
        "0": ["incorrect",],
        "1": ["correct"],
    }

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words=label_words,
    tokenizer=tokenizer,
)

wrapped_example = promptTemplate.wrap_one_example(raw_data['train'][0])
print(wrapped_example)


promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer,
    freeze_plm=True
)

data_loader = PromptDataLoader(
    dataset=raw_data['train'],
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
)

print(data_loader.tensor_dataset[0])

# promptModel.eval()
# with torch.no_grad():
#     for batch in data_loader:
#         logits = promptModel(batch)
#         preds = torch.argmax(logits, dim = -1)
#         print(classes[preds])


# logger.info(promptModel)

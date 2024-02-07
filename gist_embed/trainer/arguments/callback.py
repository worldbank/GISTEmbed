from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CallbackArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    # HuggingFace Hub
    callback_save_to_hub: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the model to the HuggingFace Hub in the ModelSaveCallback."},
    )
    callback_hub_model_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the model on the HuggingFace Hub. If None, it will be the same as output_dir."},
    )
    callback_hub_organization: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the organization on the HuggingFace Hub."},
    )
    callback_hub_private: bool = field(
        default=False,
        metadata={"help": "Whether or not the model is private on the HuggingFace Hub."},
    )
    callback_hub_exist_ok: bool = field(
        default=False,
        metadata={"help": "Whether or not to overwrite the model on the HuggingFace Hub."},
    )
    callback_hub_replace_model_card: bool = field(
        default=False,
        metadata={"help": "Whether or not to replace the model card on the HuggingFace Hub."},
    )
    callback_hub_train_datasets: Optional[list[str]] = field(
        default=None,
        metadata={"help": "The name of the datasets used to train the model on the HuggingFace Hub."},
    )
    callback_hub_run_as_future: bool = field(
        default=False,
        metadata={"help": "Whether or not to run the upload to the HuggingFace Hub as a future."},
    )

    def __post_init__(self):
        pass

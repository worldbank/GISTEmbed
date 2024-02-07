from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GISTArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    gist_loss_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The type of loss to use, defaults to contrastive. Can be contrastive, improved_contrastive, triplet_contrastive, orthogonal, hierarchical_contrastive, guided, guided-triplet, or guided-triplet-soft."
            )
        },
    )
    gist_output_dim: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The output dimension of the model. When None, it will be the same as the experts."
            )
        },
    )
    gist_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to normalize the embeddings."},
    )
    gist_denormalize_experts: bool = field(
        default=False,
        metadata={"help": "Whether to denormalize the experts. This invalidates the use of gist_normalize_experts."},
    )
    gist_schedule_cl_temperature: bool = field(
        default=False,
        metadata={"help": "Whether to schedule the contrastive loss temperature. This will overrider the cl_temperature argument."},
    )
    gist_cl_temperature_decay_rate: float = field(
        default=0.9999,
        metadata={"help": "The decay rate for the contrastive loss temperature."},
    )
    gist_cl_temperature_init: float = field(
        default=1.0,
        metadata={"help": "The initial contrastive loss temperature."},
    )
    gist_cl_temperature_min: float = field(
        default=0.001,
        metadata={"help": "The minimum contrastive loss temperature."},
    )
    gist_orthogonal_loss_margin: float = field(
        default=0.0,
        metadata={"help": "The margin for the cosine/orthogonal loss."},
    )
    gist_use_query_instruction: bool = field(
        default=False,
        metadata={"help": "Whether to use query instruction."},
    )
    gist_medi_data_name: str = field(
        default="medi-data.json",
        metadata={"help": "The name of the medi data."},
    )
    gist_hcl_num_subembeddings: int = field(
        default=1,
        metadata={"help": "The number of subembeddings for the hierarchical contrastive loss."},
    )
    gist_freeze_base_num_steps: int = field(
        default=0,
        metadata={"help": "The number of steps to freeze the base model. If 0, the base model will not be frozen."},
    )
    gist_guide_model_name_or_path: str = field(
        default=None,
        metadata={"help": "The guide model for identifying hard negatives. If this is provided, the `MixEmbGuidedTrainer` will be used."}
    )
    gist_medi_data_name_revision: str = field(
        default=None,
        metadata={"help": "The revision for the dataset if medi_data_name is from Hf Hub."}
    )
    gist_script_id: str = field(
        default=None,
        metadata={"help": "The script id is for validating the parameters."}
    )
    gist_cl_temperature: Optional[float] = field(
        default=None,
        metadata={"help": "contrastive temperature"},
    )
    gist_tl_margin: Optional[float] = field(
        default=None,
        metadata={"help": "margin for triplet loss"},
    )
    gist_auto_model_pooling: Optional[str] = field(
        default="mean",
        metadata={"help": "auto model pooling"},
    )
    gist_negative_mode: Optional[str] = field(
        default="all",
        metadata={"help": "negative mode. Can be all, hard, or hard+random"},
    )

    def __post_init__(self):
        pass


# This script is based on https://github.com/xlang-ai/instructor-embedding/blob/main/train.py
import logging
import os
import torch
import random
import sys
import json
import warnings

import datasets
import nltk  # Here to have a nice missing dependency error message early on

import transformers
from filelock import FileLock
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, HubStrategy
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from datasets import Dataset,DatasetDict
from datasets.fingerprint import Hasher

from sentence_transformers.models import Dense, Normalize
from gist_embed.base import EncoderSentenceTransformer
from gist_embed.trainer.callbacks import ModelSaveCallback, ContrastiveLossTemperatureCallback
from gist_embed.trainer import MixEmbTrainer, GISTTrainer
from gist_embed.validator import validate_script_id
from gist_embed.trainer.arguments import (
    CallbackArguments,
    DataTrainingArguments,
    ModelArguments,
    GISTArguments,
)

check_min_version("4.20.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


class DataCollatorForGIST(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        texts = {}
        for key in ["", "query", "pos", "neg"]:
            key = "texts" if key == "" else f"{key}_texts"

            if key in features[0].keys():
                texts[key] = [feature.pop(key) for feature in features]

        output = super().__call__(features, return_tensors)

        if len(texts) > 0:
            output.update(texts)

        return output


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, GISTArguments, CallbackArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, gist_args, callback_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, gist_args, callback_args = parser.parse_args_into_dataclasses()

    assert training_args.hub_strategy == HubStrategy.CHECKPOINT, "Only checkpoint strategy is supported for now."

    print(training_args)
    print(gist_args)
    assert gist_args.gist_loss_type in ("contrastive", "improved_contrastive", "triplet_contrastive", "orthogonal", "hierarchical_contrastive", "guided", "guided-triplet", "guided-triplet-soft")

    if gist_args.gist_guide_model_name_or_path == "None":
        gist_args.gist_guide_model_name_or_path = None

    data_args.output_dir = training_args.output_dir
    data_args.model_name_or_path = model_args.model_name_or_path
    data_args.tokenizer_name_or_path = model_args.model_name_or_path
    training_args.gist_cl_temperature = gist_args.gist_cl_temperature
    training_args.gist_tl_margin = gist_args.gist_tl_margin
    training_args.gist_loss_type = gist_args.gist_loss_type
    training_args.gist_orthogonal_loss_margin = gist_args.gist_orthogonal_loss_margin
    training_args.gist_router_aux_loss_coef = 0
    training_args.gist_hcl_num_subembeddings = gist_args.gist_hcl_num_subembeddings
    training_args.gist_guide_model_name_or_path = gist_args.gist_guide_model_name_or_path
    training_args.gist_guide_model_cache_dir = model_args.cache_dir
    training_args.gist_negative_mode = gist_args.gist_negative_mode
    training_args.remove_unused_columns = False

    if training_args.resume_from_checkpoint == "None":
        training_args.resume_from_checkpoint = None

    gist_args.max_source_length = data_args.max_source_length

    validate_script_id(gist_args, model_args, training_args)

    if not os.path.isdir(data_args.output_dir):
        os.makedirs(data_args.output_dir,exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.ERROR
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    set_seed(training_args.seed)

    if "/" in gist_args.gist_medi_data_name:
        # We assume the data is from the hub
        hf_data = datasets.load_dataset(gist_args.gist_medi_data_name, split="train", revision=gist_args.gist_medi_data_name_revision)
        def structure_example(example):
            return dict(
                query=[example["query_instruct"], example["query"]],
                pos=[example["pos_instruct"], example["pos"]],
                neg=[example["neg_instruct"], example["neg"]],
                task_name=example["task_name"],
            )
        train_examples_raw = hf_data.map(structure_example).to_list()
    else:
        with open(os.path.join(model_args.cache_dir, gist_args.gist_medi_data_name)) as f:
            train_examples_raw = json.load(f)

    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:data_args.debug_mode]
    old_train_examples_raw = train_examples_raw
    total_train_n = len(old_train_examples_raw)

    real_batch_size = max(training_args.per_device_train_batch_size,
                          training_args.per_device_train_batch_size * torch.cuda.device_count())
    # print('real_batch_size: ', real_batch_size,training_args.per_device_train_batch_size,torch.cuda.device_count())
    def get_examples_raw(old_examples_raw, total_n, real_batch_size):
        examples_raw = []
        for idx in range(0, total_n, real_batch_size):
            local_task_name = old_examples_raw[idx]['task_name']
            cur_batch = []
            include_batch = True
            for idx1 in range(idx, min(idx + real_batch_size, total_n)):
                if not old_examples_raw[idx1]['task_name'] == local_task_name:
                    print(f'one batch in task {old_examples_raw[idx1]["task_name"]} is skipped')
                    include_batch = False
                    break
                else:
                    cur_batch.append(old_examples_raw[idx1])
            if include_batch and len(cur_batch) == real_batch_size:
                examples_raw.append(cur_batch)
        return examples_raw

    train_examples_raw = get_examples_raw(old_train_examples_raw, total_train_n, real_batch_size)
    random.shuffle(train_examples_raw)

    if data_args.max_examples is not None and len(train_examples_raw*real_batch_size)>data_args.max_examples:
        train_examples_raw = train_examples_raw[:int(data_args.max_examples/real_batch_size)]

    train_examples_raw_batch = train_examples_raw
    train_examples_raw = []

    for b in train_examples_raw_batch:
        train_examples_raw += b
    print(f'There are {len(train_examples_raw)} pairs to train in total.')

    if data_args.debug_mode:
        train_examples_raw = train_examples_raw[:int(data_args.debug_mode)]

    get_dataset_params = dict(
        gist_use_query_instruction=gist_args.gist_use_query_instruction,
    )

    def get_dataset(examples_raw):
        # Ignore the instruction from the medi data. The medi data comes in
        # ["query": [instruction, text], "pos": [instruction, text], "neg": [instruction, text] format.
        # So we only get the last item from query, pos, and neg.
        examples = {'query':[],'pos':[],'neg':[],'task_name':[]}
        task_name_map = {}
        task_count = 0

        for cur_e in examples_raw:
            for k in ['query','pos','neg']:
                if get_dataset_params["gist_use_query_instruction"] and k == 'query':
                    v = " ".join(cur_e[k]).strip()
                else:
                    v = cur_e[k][-1].strip()
                examples[k].append(v)

            if not cur_e['task_name'] in task_name_map:
                task_name_map[cur_e['task_name']] = task_count
                task_count += 1

            examples['task_name'].append(task_name_map[cur_e['task_name']])

        return examples

    # Find a way to make get_dataset idempotent later.
    train_dataset_prefix = gist_args.gist_medi_data_name.replace('.json','').replace("-","_").replace("/", "_")
    train_dataset_dict_path = os.path.join(
        model_args.cache_dir,
        f"{train_dataset_prefix}-rbs_{real_batch_size}-train_dataset_dict-{Hasher.hash(train_examples_raw)}-{Hasher.hash(get_dataset)}")

    print("train_dataset_dict_path:", train_dataset_dict_path)

    try:
        train_raw_datasets = DatasetDict.load_from_disk(train_dataset_dict_path)
    except:
        train_raw_datasets = DatasetDict({'train': Dataset.from_dict(get_dataset(train_examples_raw))})
        train_raw_datasets.save_to_disk(train_dataset_dict_path)

        # Load so the subsequent processing also gets cached on first run.
        train_raw_datasets = DatasetDict.load_from_disk(train_dataset_dict_path)

    column_names = train_raw_datasets["train"].column_names

    tokenizer_params = {
        'padding': 'max_length',
        'truncation': 'longest_first',
        'return_tensors': 'pt',
        'max_length': data_args.max_source_length
    }

    def preprocess_function(examples):
        all_tokenized = None
        for key in ['query', 'pos', 'neg']:
            if key in examples:
                tokenized = tokenizer(examples[key], **tokenizer_params)
                tokenized["texts"] = examples[key]

                if all_tokenized is None:
                    all_tokenized = tokenized.copy()
                    for k in tokenized.keys():
                        if not isinstance(all_tokenized[k], list):
                            all_tokenized[k] = all_tokenized[k].tolist()

                for k in tokenized.keys():
                    if not isinstance(tokenized[k], list):
                        all_tokenized[f'{key}_{k}'] = tokenized[k].tolist()
                    else:
                        all_tokenized[f'{key}_{k}'] = tokenized[k]

        if 'task_name' in examples:
            all_tokenized['task_name'] = examples['task_name']

        return all_tokenized

    train_dataset = train_raw_datasets["train"]

    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    model = EncoderSentenceTransformer(
        model_args.model_name_or_path,
        cache_folder=model_args.cache_dir,
        auto_model_pooling=gist_args.gist_auto_model_pooling,
    )

    if model.max_seq_length != data_args.max_source_length:
        assert data_args.max_source_length <= model._first_module().auto_model.embeddings.position_embeddings.num_embeddings

        model.max_seq_length = data_args.max_source_length

    if gist_args.gist_output_dim is not None:
        if gist_args.gist_output_dim != model.get_sentence_embedding_dimension():
            # If the embedding dimension is not the same as the expected dimension, let's add a linear layer to
            # project the embeddings to the expected dimension.
            # Check if there is a Normalize module at the end of the model. If so, remove it.
            if model._last_module().__module__.endswith("Normalize"):
                model.pop(-1)

                assert not model._last_module().__module__.endswith("Normalize")

            model.append(
                Dense(
                    in_features=model.get_sentence_embedding_dimension(),
                    out_features=gist_args.gist_output_dim,
                    bias=False,
                )
            )

    if gist_args.gist_normalize:
        if not model._last_module().__module__.endswith("Normalize"):
            model.append(Normalize())
    else:
        if model._last_module().__module__.endswith("Normalize"):
            model.pop(-1)
            assert not model._last_module().__module__.endswith("Normalize")

    print("model:", model.device)

    data_collator = DataCollatorForGIST(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    callbacks = [
        ModelSaveCallback(
            save_dir=training_args.output_dir,
            sub_dir="snapshot",
            save_to_hub=callback_args.callback_save_to_hub,
            hub_model_name=callback_args.callback_hub_model_name,  # Use default name (same as output_dir)
            hub_organization=callback_args.callback_hub_organization,
            hub_private=callback_args.callback_hub_private,
            hub_exist_ok=callback_args.callback_hub_exist_ok,
            hub_replace_model_card=callback_args.callback_hub_replace_model_card,
            # hub_train_datasets=hub_train_datasets,
            hub_run_as_future=callback_args.callback_hub_run_as_future,
            verbose=False,
        ),
    ]

    if gist_args.gist_schedule_cl_temperature:
        callbacks.append(
            ContrastiveLossTemperatureCallback(
                temperature_init=gist_args.gist_cl_temperature_init,
                temperature_decay_rate=gist_args.gist_cl_temperature_decay_rate,
                temperature_min=gist_args.gist_cl_temperature_min,
            )
        )

    if not gist_args.gist_loss_type.startswith("guided"):
        trainer_cls = MixEmbTrainer
    else:
        warnings.warn(f"GISTTrainer is used. Arguments on loss not related to the guided mode are ignored!")
        trainer_cls = GISTTrainer

    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=None,
        callbacks=callbacks,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.save(training_args.output_dir)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()

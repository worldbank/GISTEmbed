import os
import numpy as np
from typing import List, Union, Optional
import torch
import json
import os
from collections import OrderedDict
import sentence_transformers
from sentence_transformers.util import import_from_string, batch_to_device

from sentence_transformers import __version__, SentenceTransformer
from sentence_transformers.models import Transformer, Pooling, Normalize
from sentence_transformers.util import load_dir_path, load_file_path
from tqdm import trange
import logging
import shutil
import warnings

from .util import save_to_hub

logger = logging.getLogger(__name__)


class EncoderSentenceTransformer(SentenceTransformer):
    def __init__(self, *args, **kwargs):

        # Add support for controllable feature generation in the clustering tasks.
        self.task = None
        self.normed_tasks = []

        if "normed_tasks" in kwargs:
            self.normed_tasks = kwargs.pop("normed_tasks")
            assert self.normed_tasks is not None

            if isinstance(self.normed_tasks, str):
                self.normed_tasks = [self.normed_tasks]

        self.concat_norm = False  # If True, this overrides the encoding output so that it returns the concatenation of the unnormalized and the normalized embeddings.
        if "concat_norm" in kwargs:
            self.concat_norm = kwargs.pop("concat_norm")

        self.auto_model_pooling = "mean"
        if "auto_model_pooling" in kwargs:
            self.auto_model_pooling = kwargs.pop("auto_model_pooling")

        if "overwrite" in kwargs:
            if kwargs.pop("overwrite") and kwargs.get("cache_folder"):
                model_name_or_path = kwargs.get("model_name_or_path", args[0])
                model_id = model_name_or_path.replace("/", "_")
                path = os.path.join(kwargs.get("cache_folder"), model_id)

                if os.path.exists(path):
                    shutil.rmtree(path)

        super().__init__(*args, **kwargs)


        if self.normed_tasks and self._last_module().__module__.endswith(".Normalize"):
            warnings.warn("The normalization module for this model will be removed since `normed_tasks` is not empty.")
            self.pop(-1)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self._first_module().auto_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self._first_module().auto_model.gradient_checkpointing_disable()

    def _load_auto_model(
        self,
        model_name_or_path: str,
        token: Optional[Union[bool, str]],
        cache_folder: Optional[str],
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Creates a simple Transformer + Mean Pooling model and returns the modules
        """
        logger.warning(
            "No sentence-transformers model found with name {}. Creating a new one with {} pooling.".format(
                model_name_or_path,
                self.auto_model_pooling.upper(),
            )
        )
        transformer_model = Transformer(
            model_name_or_path,
            cache_dir=cache_folder,
            model_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
            tokenizer_args={"token": token, "trust_remote_code": trust_remote_code, "revision": revision},
        )
        pooling_model = Pooling(transformer_model.get_word_embedding_dimension(), pooling_mode=self.auto_model_pooling)

        return [transformer_model, pooling_model]

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False):
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        self.eval()
        # if show_progress_bar is None:
        #     show_progress_bar = (logger.getEffectiveLevel()==logging.INFO or logger.getEffectiveLevel()==logging.DEBUG)
        init_state_norm = self._last_module().__module__.endswith(".Normalize")
        if self.concat_norm and init_state_norm:
            self.pop(-1)
            assert not self._last_module().__module__.endswith(".Normalize")

        if self.normed_tasks and self.task in self.normed_tasks:
            if not init_state_norm:
                warnings.warn("Forcing embedding normalization")
                self.append(Normalize())
        else:
            if init_state_norm:
                self.pop(-1)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        if device is None:
            device = self.device

        self.to(device)

        all_embeddings = []
        if isinstance(sentences[0],list):
            lengths = []
            for sen in sentences:
                lengths.append(-self._text_length(sen[1]))
            length_sorted_idx = np.argsort(lengths)
        else:
            length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)

            with torch.no_grad():
                out_features = self.forward(features)

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                elif output_value is None:  #Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['sentence_embedding'])):
                        row =  {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:   #Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings and not self.concat_norm:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)

            if self.concat_norm:
                all_embeddings = torch.cat((all_embeddings, torch.nn.functional.normalize(all_embeddings, p=2, dim=1)), dim=1)

        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
            if self.concat_norm:
                all_embeddings = np.hstack((all_embeddings, all_embeddings / np.linalg.norm(all_embeddings, ord=2, axis=1, keepdims=True)))

        if input_was_string:
            all_embeddings = all_embeddings[0]

        # # Put the Normalize module back if it was initially part
        # # of the model.
        # if self.concat_norm and init_state_norm:
        #     self.append(Normalize())

        return all_embeddings

    def module_select(self, module_config):
        module_class = import_from_string(module_config["type"])

        return module_class

    def _load_sbert_model(
        self,
        model_name_or_path: str,
        token: Optional[Union[bool, str]],
        cache_folder: Optional[str],
        revision: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        """
        Loads a full sentence-transformers model
        """
        # Check if the config_sentence_transformers.json file exists (exists since v2 of the framework)
        config_sentence_transformers_json_path = load_file_path(
            model_name_or_path,
            "config_sentence_transformers.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
        )

        # Load this so we can use for the evaluation logs.
        self.commit_info_json_path = load_file_path(
            model_name_or_path,
            "commit-info.json",
            token=token,
            cache_folder=cache_folder,
            revision=revision,
        )

        if config_sentence_transformers_json_path is not None:
            with open(config_sentence_transformers_json_path) as fIn:
                self._model_config = json.load(fIn)

            if (
                "__version__" in self._model_config
                and "sentence_transformers" in self._model_config["__version__"]
                and self._model_config["__version__"]["sentence_transformers"] > __version__
            ):
                logger.warning(
                    "You try to use a model that was created with version {}, however, your version is {}. This might cause unexpected behavior or errors. In that case, try to update to the latest version.\n\n\n".format(
                        self._model_config["__version__"]["sentence_transformers"], __version__
                    )
                )

        # Check if a readme exists
        model_card_path = load_file_path(
            model_name_or_path, "README.md", token=token, cache_folder=cache_folder, revision=revision
        )
        if model_card_path is not None:
            try:
                with open(model_card_path, encoding="utf8") as fIn:
                    self._model_card_text = fIn.read()
            except Exception:
                pass

        # Load the modules of sentence transformer
        modules_json_path = load_file_path(
            model_name_or_path, "modules.json", token=token, cache_folder=cache_folder, revision=revision
        )
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)

        modules = OrderedDict()
        for module_config in modules_config:
            module_class = self.module_select(module_config)
            # For Transformer, don't load the full directory, rely on `transformers` instead
            # But, do load the config file first.
            if module_class == Transformer and module_config["path"] == "":
                kwargs = {}
                for config_name in [
                    "sentence_bert_config.json",
                    "sentence_roberta_config.json",
                    "sentence_distilbert_config.json",
                    "sentence_camembert_config.json",
                    "sentence_albert_config.json",
                    "sentence_xlm-roberta_config.json",
                    "sentence_xlnet_config.json",
                ]:
                    config_path = load_file_path(
                        model_name_or_path, config_name, token=token, cache_folder=cache_folder, revision=revision
                    )
                    if config_path is not None:
                        with open(config_path) as fIn:
                            kwargs = json.load(fIn)
                        break
                hub_kwargs = {"token": token, "trust_remote_code": trust_remote_code, "revision": revision}
                if "model_args" in kwargs:
                    kwargs["model_args"].update(hub_kwargs)
                else:
                    kwargs["model_args"] = hub_kwargs
                if "tokenizer_args" in kwargs:
                    kwargs["tokenizer_args"].update(hub_kwargs)
                else:
                    kwargs["tokenizer_args"] = hub_kwargs
                module = Transformer(model_name_or_path, cache_dir=cache_folder, **kwargs)
            else:
                module_path = load_dir_path(
                    model_name_or_path,
                    module_config["path"],
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                )
                module = module_class.load(module_path)
            modules[module_config["name"]] = module

        return modules

    def save_to_hub(self,
                    repo_id: str,
                    organization: Optional[str] = None,
                    token: Optional[str] = None,
                    private: Optional[bool] = None,
                    commit_message: str = "Add new SentenceTransformer model.",
                    local_model_path: Optional[str] = None,
                    exist_ok: bool = False,
                    replace_model_card: bool = False,
                    train_datasets: Optional[List[str]] = None,
                    run_as_future: bool = False,
                ):
        """
        Uploads all elements of this Sentence Transformer to a new HuggingFace Hub repository.

        :param repo_id: Repository name for your model in the Hub, including the user or organization.
        :param token: An authentication token (See https://huggingface.co/settings/token)
        :param private: Set to true, for hosting a private model
        :param commit_message: Message to commit while pushing.
        :param local_model_path: Path of the model locally. If set, this file path will be uploaded. Otherwise, the current model will be uploaded
        :param exist_ok: If true, saving to an existing repository is OK. If false, saving only to a new repository is possible
        :param replace_model_card: If true, replace an existing model card in the hub with the automatically created model card
        :param train_datasets: Datasets used to train the model. If set, the datasets will be added to the model card in the Hub.
        :param organization: Deprecated. Organization in which you want to push your model or tokenizer (you must be a member of this organization).

        :return: The url of the commit of your model in the repository on the Hugging Face Hub.
        """
        save_to_hub(self, repo_id, organization, token, private, commit_message, local_model_path, exist_ok, replace_model_card, train_datasets, run_as_future)

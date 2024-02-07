import os
import json
import torch
from pathlib import Path
from transformers.trainer_callback import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import PushInProgress
from huggingface_hub import create_repo, upload_folder
from logging import getLogger
import shutil

logger = getLogger(__name__)

from torch import nn


class ModelSaveCallback(TrainerCallback):
    def __init__(self, save_dir: str, sub_dir: str = "snapshot", save_to_hub: bool = False, hub_model_name: str = None, hub_organization: str = None, hub_private: bool = False, hub_exist_ok: bool = False, hub_replace_model_card: bool = False, hub_train_datasets: list[str] = None, hub_run_as_future: bool = True, verbose: bool = True):
        self._output_dir = save_dir
        self._save_dir = os.path.join(save_dir, sub_dir)
        self._latest_dir = os.path.join(self._save_dir, "latest")
        self._best_dir = os.path.join(self._save_dir, "best")
        self._verbose = verbose
        self._best_loss = None
        self.push_in_progress = None

        assert hub_organization is not None or not save_to_hub, "Must provide a hub_organization representing the username or organization to save to HuggingFace Hub"

        # HuggingFace Hub
        self._save_to_hub = save_to_hub
        self._hub_model_name = hub_model_name or os.path.basename(save_dir.rstrip("/"))
        self._hub_train_datasets = hub_train_datasets
        self._hub_exist_ok = hub_exist_ok
        self._hub_private = hub_private
        self._hub_organization = hub_organization
        self._hub_replace_model_card = hub_replace_model_card
        self._hub_run_as_future = hub_run_as_future

    def build_commit_message(self, state: TrainerState, model: nn.Module = None):
        # Get the last log history and build a commit message
        log_history = state.log_history[-1]
        # commit_message = f"Training for {log_history['epoch']} epochs, {log_history['step']} steps, {log_history['loss']} loss, {log_history['learning_rate']} learning rate."

        if model is not None:
            try:
                log_history = {**log_history, "gate_temperature": model._last_module().gate_temperature}
            except:
                pass

        commit_message = json.dumps(log_history)
        return commit_message

    def save_and_upload(self, model: nn.Module, save_dir: str, state: TrainerState):
        model.save(save_dir)

        if self._save_to_hub:
            if save_dir == self._best_dir:
                repo_name = f"{self._hub_model_name}-best"
                commit_message = self.build_commit_message(state, model)
            elif save_dir == self._latest_dir:
                repo_name = f"{self._hub_model_name}-latest"
                commit_message = self.build_commit_message(state, model)
            else:
                raise ValueError(f"Unknown save_dir: {save_dir}")

            repo_id = f"{self._hub_organization}/{repo_name}"

            model.save_to_hub(
                repo_id=repo_id,
                commit_message=commit_message,
                private=self._hub_private,
                exist_ok=self._hub_exist_ok,
                replace_model_card=self._hub_replace_model_card,
                train_datasets=self._hub_train_datasets,
                run_as_future=self._hub_run_as_future,
            )

    def save_checkpoint(self, model: nn.Module, state: TrainerState):
        repo_name = f"{self._hub_model_name}-checkpoint"
        commit_message = self.build_commit_message(state, model)

        repo_url = create_repo(repo_name, private=self._hub_private, exist_ok=True)
        self.hub_model_id = repo_url.repo_id
        self.push_in_progress = None

        # Get the most recent checkpoint
        cpt = sorted(Path(self._output_dir).glob("checkpoint-*"), key=os.path.getctime)
        if len(cpt) > 0:
            checkpoint_folder = str(cpt[-1].absolute())
        else:
            return

        dest = os.path.join(os.path.dirname(checkpoint_folder), "latest-checkpoint")
        checkpoint_folder = shutil.copytree(checkpoint_folder, dest, dirs_exist_ok=True)

        checkpoint_push = upload_folder(
            repo_id=repo_url.repo_id,
            path_in_repo="latest-checkpoint",
            folder_path=checkpoint_folder,
            commit_message=commit_message,
            run_as_future=True,
        )

        push_jobs = [checkpoint_push]

        if self.push_in_progress is None:
            self.push_in_progress = PushInProgress(push_jobs)
        else:
            self.push_in_progress.jobs.extend(push_jobs)

        # logger.info(f"Saving checkpoint {checkpoint_folder}...")

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module = None, **kwargs):
        if self._verbose:
            print("Saving model snapshot...")

        if self._best_loss is None:
            if len(state.log_history) > 0:
                self._best_loss = state.log_history[-1]["loss"]
                self.save_and_upload(model, self._best_dir, state)

        if self._best_loss > state.log_history[-1]["loss"]:
            if self._verbose:
                print("Saving best model snapshot...")

            self._best_loss = state.log_history[-1]["loss"]
            self.save_and_upload(model, self._best_dir, state)

        # Save the latest model snapshot
        self.save_and_upload(model, self._latest_dir, state)

        # Get most recent checkpoint and upload to HuggingFace Hub
        self.save_checkpoint(model, state)

        # Free up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class GateAnnealingCallback(TrainerCallback):
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module = None, **kwargs):
        # This assumes that the last module is the gate module.
        model._last_module().current_train_step = state.global_step



class DetachExpertsCallback(TrainerCallback):
    def __init__(self, detach_loss_threshold: float = 0):
        self.detach_loss_threshold = detach_loss_threshold

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module = None, **kwargs):
        # This assumes that the last module is the gate module.

        if len(state.log_history) == 0 or "loss" not in state.log_history[-1]:
            return

        if self.detach_loss_threshold > 0:
            model._last_module().detach_experts = state.log_history[-1]["loss"] > self.detach_loss_threshold



class ContrastiveLossTemperatureCallback(TrainerCallback):
    def __init__(self, temperature_init: float = 1.0, temperature_decay_rate: float = 0.9999, temperature_min: float = 0.001):
        # The default values will decay the temperature from 1 to 0.01, and every 1000 steps the temperature will have been decayed by 0.01.
        self.temperature_init = temperature_init
        self.temperature_decay_rate = temperature_decay_rate
        self.temperature_min = temperature_min

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        args.gist_cl_temperature = min(max(self.temperature_min, self.temperature_decay_rate ** state.global_step), self.temperature_init)

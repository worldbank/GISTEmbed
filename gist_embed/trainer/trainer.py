import random
import torch
from transformers import (
    Seq2SeqTrainer
)
from torch.utils.data import SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .loss import (
    in_batch_contrastive_loss,
    improved_in_batch_contrastive_loss,
    triplet_constrastive_loss,
    orthogonal_loss,
    hierarchical_contrastive_loss,
    guided_in_batch_contrastive_loss,
)
from ..base import EncoderSentenceTransformer


def has_length(dataset):
    """
    Checks if the dataset implements __len__() and it doesn't raise an error
    """
    try:
        return len(dataset) is not None
    except TypeError:
        # TypeError: len() of unsized object
        return False


class MixEmbTrainer(Seq2SeqTrainer):
    def _get_train_sampler(self) :
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            return SequentialSampler(self.train_dataset)
        else:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.args.gist_loss_type in ("contrastive", "default"):
            output = in_batch_contrastive_loss(
                args=self.args,
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
                mode="vectorized",
            )
        elif self.args.gist_loss_type == "improved_contrastive":
            output = improved_in_batch_contrastive_loss(
                args=self.args,
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )
        elif self.args.gist_loss_type == "triplet_contrastive":
            output = triplet_constrastive_loss(
                args=self.args,
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )
        elif self.args.gist_loss_type == "orthogonal":
            output = orthogonal_loss(
                args=self.args,
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )
        elif self.args.gist_loss_type == "hierarchical_contrastive":
            output = hierarchical_contrastive_loss(
                args=self.args,
                model=model,
                inputs=inputs,
                return_outputs=return_outputs,
            )
        else:
            raise ValueError(f"Invalid loss type: {self.args.gist_loss_type}. Should be either 'contrastive', 'improved_contrastive', 'triplet_contrastive', 'orthogonal', or 'hierarchical_contrastive'.")

        return output


class StochasticTrainer(MixEmbTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """This creates a scheduler that returns some random learning rate between self.args.min_learning_rate and self.args.learning_rate."""

        if self.args.min_learning_rate is not None:
            min_lr = self.args.min_learning_rate
        else:
            min_lr = 1 / num_training_steps

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: random.uniform(min_lr, 1),
            last_epoch=-1,
            verbose=False
        )

        self._created_lr_scheduler = True

        return self.lr_scheduler


class GISTTrainer(MixEmbTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.args.gist_guide_model_name_or_path is None:
            self.guide = None
        else:
            self.guide = EncoderSentenceTransformer(self.args.gist_guide_model_name_or_path, cache_folder=self.args.gist_guide_model_cache_dir)

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.guide:
            if self.guide.device != model.device:
                self.guide.to(model.device)

        return guided_in_batch_contrastive_loss(
            args=self.args,
            guide=self.guide,
            model=model,
            inputs=inputs,
            return_outputs=return_outputs,
        )

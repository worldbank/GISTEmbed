# GISTEmbed

The GISTEmbed framework (Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning) introduces an innovative approach to dynamically mine training negatives within a batch, serving as contrastive samples for fine-tuning embedding models. At the core of GISTEmbed is the utilization of a guide model, which assesses the relevance of samples in the batch against a query-positive pair. This model ensures that only examples deemed irrelevant are selected as training negatives.

This methodology is particularly advantageous for fine-tuning smaller models, leading to notable improvements across a wide range of NLP tasks. By focusing on the in-sample selection of negatives, GISTEmbed addresses common challenges in contrastive learning, such as the efficient and effective identification of informative negative samples.

Compared to traditional methods, which often rely on random or heuristic-based selection, GISTEmbed's guided approach ensures a higher quality of training negatives, contributing to more robust and generalizable embeddings.

<br>
<br>
<p align="center">
<img src="https://github.com/avsolatorio/GISTEmbed/raw/main/img/GISTEmbed%20Model.png" style="width:75%"/>
</p>
<p align="center">
<strong>GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning</strong>
<br>
<a href="https://arxiv.org/abs/2402.16829" target="_blank">Paper on ArXiv</a>
</p>
<br>


The model does not require any instruction for generating embeddings. This means that queries for retrieval tasks can be directly encoded without crafting instructions.

# Trained models

We have fine-tuned various models using the GISTEmbed framework. The models are available on the Hugging Face model hub:

- [avsolatorio/GIST-large-Embedding-v0](https://huggingface.co/avsolatorio/GIST-large-Embedding-v0): The model fine-tuned using the GISTEmbed framework and the MEDI+MTEBcls dataset. The base model used is the [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5).
- [avsolatorio/GIST-Embedding-v0](https://huggingface.co/avsolatorio/GIST-Embedding-v0): The model fine-tuned using the GISTEmbed framework and the MEDI+MTEBcls dataset. The base model used is the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5).
- [avsolatorio/GIST-small-Embedding-v0](https://huggingface.co/avsolatorio/GIST-small-Embedding-v0): The model fine-tuned using the GISTEmbed framework and the MEDI+MTEBcls dataset. The base model used is the [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).
- [avsolatorio/GIST-all-MiniLM-L6-v2](https://huggingface.co/avsolatorio/GIST-all-MiniLM-L6-v2): The model fine-tuned using the GISTEmbed framework and the MEDI+MTEBcls dataset. The base model used is the [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).


# Data

The dataset used is a compilation of the MEDI dataset and the MTEB Classification training dataset. Third-party datasets may be subject to additional terms and conditions under their associated licenses. A HuggingFace Dataset version of the compiled dataset, and the specific revision used to train the model, is available:

- Dataset: [avsolatorio/medi-data-mteb_avs_triplets](https://huggingface.co/datasets/avsolatorio/medi-data-mteb_avs_triplets)
- Revision: 238a0499b6e6b690cc64ea56fde8461daa8341bb

The dataset contains a `task_type` key which can be used to select only the mteb classification tasks (prefixed with `mteb_`).

The **MEDI Dataset** is published in the following paper: [One Embedder, Any Task: Instruction-Finetuned Text Embeddings](https://arxiv.org/abs/2212.09741).

The MTEB Benchmark results of the GIST embedding model, compared with the base model, suggest that the fine-tuning dataset has perturbed the model considerably, which resulted in significant improvements in certain tasks while adversely degrading performance in some.

The retrieval performance for the TRECCOVID task is of note. The fine-tuning dataset does not contain significant knowledge about COVID, which could have caused the observed performance degradation. We found some evidence, detailed in the paper, that thematic coverage of the fine-tuning data can affect downstream performance.

# Usage

The model can be easily loaded using the Sentence Transformers library.

```Python
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

revision = None  # Replace with the specific revision to ensure reproducibility in  case the model is updated.

model = SentenceTransformer("avsolatorio/GIST-Embedding-v0", revision=revision)

texts = [
    "Illustration of the REaLTabFormer model. The left block shows the non-relational tabular data model using GPT-2 with a causal LM head. In contrast, the right block shows how a relational dataset's child table is modeled using a sequence-to-sequence (Seq2Seq) model. The Seq2Seq model uses the observations in the parent table to condition the generation of the observations in the child table. The trained GPT-2 model on the parent table, with weights frozen, is also used as the encoder in the Seq2Seq model.",
    "Predicting human mobility holds significant practical value, with applications ranging from enhancing disaster risk planning to simulating epidemic spread. In this paper, we present the GeoFormer, a decoder-only transformer model adapted from the GPT architecture to forecast human mobility.",
    "As the economies of Southeast Asia continue adopting digital technologies, policy makers increasingly ask how to prepare the workforce for emerging labor demands. However, little is known about the skills that workers need to adapt to these changes"
]

# Compute embeddings
embeddings = model.encode(texts, convert_to_tensor=True)

# Compute cosine-similarity for each pair of sentences
scores = F.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)

print(scores.cpu().numpy())
```

# Guided in-batch constrastive loss

For anyone interested in the technical implementation of GISTEmbed as a training mechanism, please refer to the loss computation implemented in the loss function [`guided_in_batch_contrastive_loss`](https://github.com/avsolatorio/GISTEmbed/blob/538e3d749b1944e8362c5566385111763866fa4c/gist_embed/trainer/loss.py#L599).

This loss function is subsequently used in the [`GISTTrainer`](https://github.com/avsolatorio/GISTEmbed/blob/538e3d749b1944e8362c5566385111763866fa4c/gist_embed/trainer/trainer.py#L127).


# Reproducibility

This section outlines how to fine-tune models using the GISTEmbed framework. The following steps are necessary to reproduce the results:


First, create a new conda environment and install poetry.

```
conda create -n GISTEmbed python=3.10

conda activate GISTEmbed

pip install poetry
```

Next, clone the repository and install the dependencies.

```
git clone https://github.com/avsolatorio/GISTEmbed.git

cd GISTEmbed

poetry install
```

To reduce the likelihood of encountering issues and unexpected training runs, we set up a convention that would validate the intended parameters and configurations.

One can refer to the [gist_embed/validator.py](gist_embed/validator.py) file to see the validation logic. Additional configurations must be registered in the validator to ensure that the intended parameters are correctly set.

After registering the intended configurations, an experiment script can be created to fine-tune the model. See example: [experiments/01-600-11-1-2-2-0-0-cls-normed-384-512_run_finetune_experiment.sh](experiments/01-600-11-1-2-2-0-0-cls-normed-384-512_run_finetune_experiment.sh).

Details of the arguments used in the script can be found in the [gist_embed/trainer/arguments](gist_embed/trainer/arguments) file.

To run the experiment, simply execute the following command:

```
bash experiments/01-600-11-1-2-2-0-0-cls-normed-384-512_run_finetune_experiment.sh
```

The script will execute the experiment and save the model to the specified output directory. There are configurations in the script that handles the model checkpointing to Hugging Face model hub. Ensure to change the `--callback_hub_organization <organization>` to the appropriate organization.

The script also uses WANDB for logging. Ensure to set the `WANDB_API_KEY` environment variable to enable logging to WANDB.

# Base model

We have implemented some tricks on top of the (excellent!) Sentence Transformers library to support the GISTEmbed framework. One notable trick is supporting gradient checkpointing for training the models. This is particularly useful for training large models with limited GPU memory.

See the [gist_embed/base.py](gist_embed/base.py) file for the implementation details.

# Training Parameters

Below are the training parameters used to fine-tune the model:

```
Epochs = 80
Warmup ratio = 0.1
Learning rate = 5e-6
Batch size = 32
Checkpoint step = 103500
Contrastive loss temperature = 0.01
```

Specific training details and strategies will be published shortly.

# Evaluation

The model was evaluated using the [MTEB Evaluation](https://huggingface.co/mteb) suite.


# Citation
Please cite our work if you use GISTEmbed or the datasets we published in your projects or research

```
@article{solatorio2024gistembed,
    title={GISTEmbed: Guided In-sample Selection of Training Negatives for Text Embedding Fine-tuning},
    author={Aivin V. Solatorio},
    journal={arXiv preprint arXiv:2402.16829},
    year={2024},
    URL={https://arxiv.org/abs/2402.16829}
    eprint={2402.16829},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

# Acknowledgements

This work is supported by the "KCP IV - Exploring Data Use in the Development Economics Literature using Large Language Models (AI and LLMs)" project funded by the [Knowledge for Change Program (KCP)](https://www.worldbank.org/en/programs/knowledge-for-change) of the World Bank - RA-P503405-RESE-TF0C3444.

The findings, interpretations, and conclusions expressed in this material are entirely those of the authors. They do not necessarily represent the views of the International Bank for Reconstruction and Development/World Bank and its affiliated organizations, or those of the Executive Directors of the World Bank or the governments they represent.

We also send 🤗 to the HuggingFace 🤗, Sentence Transformers, PyTorch, and to all open-sourced projects for all the open-sourced software they release.

# We use various embedding models to assess the consistency of the MEDI data.
# Given a model, we compute the embeddings for the query, positive, and negative
# samples. We then compute the cosine similarity between the query and positive
# samples, and the query and negative samples.

# We define a record consistency score as the majority vote of the selected
# embedding models. That is, if the majority of the models agree that the
# query and positive samples are more similar than the query and negative
# samples, then the record is considered consistent. Otherwise, it is
# considered inconsistent.

# We limit the models to those scoring at least 60% in the MTEB dataset
# and model size less than 0.5 GB.

# The current models are:
# 1. khoa-klaytn/bge-base-en-v1.5-angle  # 0.22 GB
# 2. BAAI/bge-base-en-v1.5  # 0.44 GB
# 3. infgrad/stella-base-en-v2  # 0.22 GB
# 4. thenlper/gte-base  # 0.22 GB
# 5. BAAI/bge-small-en-v1.5  # 0.13 GB
# 6. khoa-klaytn/bge-small-en-v1.5-angle  # 0.07 GB
# 7. intfloat/e5-base-v2  # 0.44 GB
# 8. thenlper/gte-small  # 0.07 GB
# 9. intfloat/e5-base  # 0.44 GB


# CUDA:0
model_ids_cuda0 = [
    "khoa-klaytn/bge-small-en-v1.5-angle",  # 0.07 GB
    # "thenlper/gte-small",  # 0.07 GB
    "BAAI/bge-small-en-v1.5",  # 0.13 GB
    # "infgrad/stella-base-en-v2",  # 0.22 GB
    "khoa-klaytn/bge-base-en-v1.5-angle",  # 0.22 GB
    # "thenlper/gte-base",  # 0.22 GB
    "BAAI/bge-base-en-v1.5",  # 0.44 GB
    # "intfloat/e5-base-v2",  # 0.44 GB
    "intfloat/e5-base",  # 0.44 GB
]

# CUDA:1
model_ids_cuda1 = [
    # "khoa-klaytn/bge-small-en-v1.5-angle",  # 0.07 GB
    "thenlper/gte-small",  # 0.07 GB
    # "BAAI/bge-small-en-v1.5",  # 0.13 GB
    "infgrad/stella-base-en-v2",  # 0.22 GB
    # "khoa-klaytn/bge-base-en-v1.5-angle",  # 0.22 GB
    "thenlper/gte-base",  # 0.22 GB
    # "BAAI/bge-base-en-v1.5",  # 0.44 GB
    "intfloat/e5-base-v2",  # 0.44 GB
    # "intfloat/e5-base",  # 0.44 GB
]

model_ids = model_ids_cuda0 + model_ids_cuda1


from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from collections import defaultdict
from tqdm.auto import tqdm
import gc
import torch
from gist_embed.base import EncoderSentenceTransformer


# WhereIsAI/UAE-Large-V1
def get_inconsistent(model, medi_batch, start_idx, model_batch_size=32, task_name=None):
    """Returns the indices of the inconsistent records in the batch.

    Args:
        model (SentenceTransformer): The embedding model.
        batch (list): A list of query, positive, and negative samples.

    Returns:
        list: The indices of the inconsistent records in the batch.
    """
    batch = [(o["query"][1], o["pos"][1], o["neg"][1]) for o in medi_batch]
    batch_ = [i for o in batch for i in o]

    embeddings = model.encode(batch_, batch_size=model_batch_size, show_progress_bar=True, normalize_embeddings=True)
    query = embeddings[::3]
    positive = embeddings[1::3]
    negative = embeddings[2::3]

    qp: np.ndarray = np.einsum("ij,ij->i", query, positive, dtype=float).round(4)
    qn: np.ndarray = np.einsum("ij,ij->i", query, negative, dtype=float).round(4)

    qp_mean = np.round(qp.mean(), 4)
    qn_mean = np.round(qn.mean(), 4)
    qp_std = np.round(qp.std(), 4)
    qn_std = np.round(qn.std(), 4)

    output = dict(
        start_idx=start_idx,
        inconsistent=[(i + start_idx, p, n) for i, (p, n) in enumerate(zip(qp, qn)) if p < n],
        qp_mean=qp_mean,
        qp_std=qp_std,
        qn_mean=qn_mean,
        qn_std=qn_std,
        size=len(batch),
        task_name=task_name,
    )
    output["rate"] = len(output["inconsistent"]) / output["size"]

    return output


def get_inconsistent_medi_for_model(model_id: str, medi, medi_batch_size=1000, model_batch_size=64, save_path=None, cache_folder="./cache_dir", device="mps", auto_model_pooling="mean", revision=None):
    """Returns the indices of the inconsistent records in the MEDI dataset.

    Args:
        model_id (str): The model ID.
        medi (list): The MEDI dataset.

    Returns:
        dict: The indices of the inconsistent records in the MEDI dataset.
    """
    model = EncoderSentenceTransformer(
        model_id,
        cache_folder=cache_folder,
        device=device,
        auto_model_pooling=auto_model_pooling,
        revision=revision,
    )

    if save_path is not None:
        model_id = model_id.replace("/", "_")
        suffix = "" if auto_model_pooling == "mean" else f"-{auto_model_pooling}"
        save_fname = os.path.join(save_path, f"{model_id}{suffix}.jsonl")

        # Crrate the directory if it does not exist
        os.makedirs(save_path, exist_ok=True)

    curr_task = None
    medi_inconsistent_batch = None
    medi_batch = []
    start_idx = 0
    for i, m in tqdm(enumerate(medi), desc="MEDI record"):
        if curr_task != m["task_name"]:
            print(f"Previous task: {curr_task}, current task: {m['task_name']}")
            if len(medi_batch) > 0:
                # We have reached the end of the current task
                medi_inconsistent_batch = get_inconsistent(model, medi_batch, start_idx=start_idx, model_batch_size=model_batch_size, task_name=curr_task)
            medi_batch = [m]
            start_idx = i
            curr_task = m["task_name"]
        else:
            medi_batch.append(m)

        if len(medi_batch) == medi_batch_size:
            medi_inconsistent_batch = get_inconsistent(model, medi_batch, start_idx=start_idx, model_batch_size=model_batch_size, task_name=curr_task)
            medi_batch = []
            curr_task = f"{curr_task} continued..."

        if medi_inconsistent_batch is not None:
            # Write as JSON lines, append or create
            if save_path is not None:
                with open(save_fname, "a+") as f:
                    f.write(json.dumps(medi_inconsistent_batch) + "\n")

            del medi_inconsistent_batch
            gc.collect()

            try:
                torch.mps.empty_cache()
            except:
                pass

            try:
                torch.cuda.empty_cache()
            except:
                pass

            medi_inconsistent_batch = None

    if len(medi_batch) > 0:
        medi_inconsistent_batch = get_inconsistent(model, medi_batch, start_idx=start_idx, model_batch_size=model_batch_size, task_name=curr_task)

        if medi_inconsistent_batch is not None:
            # Write as JSON lines, append or create
            if save_path is not None:
                with open(save_fname, "a+") as f:
                    f.write(json.dumps(medi_inconsistent_batch) + "\n")

            del medi_inconsistent_batch
            gc.collect()

            try:
                torch.mps.empty_cache()
            except:
                pass

            try:
                torch.cuda.empty_cache()
            except:
                pass

            medi_inconsistent_batch = None

# medi = json.load(open("medi-data/medi-data-sorted.json", "r"))

# for model_id in model_ids_cuda0:
#     print(model_id)
#     l = get_inconsistent_medi_for_model(model_id, medi, medi_batch_size=10000, model_batch_size=128, save_path="medi-task-embeddings/", device="cuda")


# for model_id in model_ids_cuda1:
#     print(model_id)
#     l = get_inconsistent_medi_for_model(model_id, medi, medi_batch_size=10000, model_batch_size=128, save_path="medi-task-embeddings/", device="cuda")



# def format_medi_record(m):
#     query = " ".join(m["query"])
#     pos = m["pos"][1]
#     neg = m["neg"][1]

#     return f"Carefully read the instruction and anchor passage and understand what it means: '{query}'\n\nWhich of the options correctly satisfies the instruction given the anchor passage?\n\nOption 1: '{pos}'\n\nOption 2: '{neg}'\n\nOption 3: None of the two options is relevant to the anchor.\n\nOnly return the option number, e.g., Option 1 if option 1 is relevant, Option 2 if option 2 is relevant, or Option 3 if none of the two is relevant."


def filter_medi_data(medi_fname, inconsistent_fname, save_path=None):
    """Filters the MEDI dataset to only include the inconsistent records.

    Args:
        medi (list): The MEDI dataset.
        inconsistent_fname (str): The path to the JSON lines file containing the indices of the inconsistent records in the MEDI dataset.
        save_path (str, optional): The path to save the filtered MEDI dataset. Defaults to None.

    Returns:
        list: The filtered MEDI dataset.
    """
    import pandas as pd
    from pathlib import Path

    medi_fname = Path(medi_fname)
    inconsistent_fname = Path(inconsistent_fname)

    medi = json.load(open(medi_fname, "r"))

    df = pd.read_json(inconsistent_fname, lines=True)

    # Get the indices of the inconsistent records
    ids = set(df["inconsistent"].map(lambda x: [i[0] for i in x]).sum())

    # Filter the MEDI dataset
    filtered_medi = [m for i, m in enumerate(medi) if i not in ids]

    print(f"Original MEDI size: {len(medi)}, filtered MEDI size: {len(filtered_medi)}")

    if save_path is None:
        save_path = medi_fname.parent / f"{medi_fname.stem}_{inconsistent_fname.stem}.json"

    with open(save_path, "w") as f:
        json.dump(filtered_medi, f)

    return save_path

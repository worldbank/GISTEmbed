
"""
Classification MTEB datasets
# - avsolatorio/covid-bing-query-gpt4
"""
import json
import os
import torch
import torch.nn as nn
from gist_embed.base import EncoderSentenceTransformer
from tqdm.auto import tqdm
import datasets
# Import set_seed
from transformers import set_seed
import fire


def batch_sim(similarity_fct, embeddings, batch_size):
    sq1 = embeddings.unsqueeze(1)
    sims_list = []

    for start_index in tqdm(range(0, len(embeddings), batch_size)):
        sims_list.append(
            similarity_fct(sq1, embeddings[start_index:start_index+batch_size].unsqueeze(0)).cpu()
        )

        try:
            torch.cuda.empty_cache()
        except:
            try:
                torch.mps.empty_cache()
            except:
                pass

    # sims = torch.cat(sims_list, dim=1).to(embeddings.device)
    sims = torch.cat(sims_list, dim=1).cpu()

    return sims


def mine_COVID(data_name: str, guide_model_name_or_path: str, cache_folder: str, save_dir: str, revision: str = None, lang: str = "en", topk: int = 100, temperature: float = 0.05, device: str = None, seed: int = 1029, negative_sampling: str = "uniform", neg_topk: int = None, use_cpu: bool = False, encode_batch_size: int = 64, return_data: bool = False, sim_batch_size: int = 32):
    assert negative_sampling in ["uniform", "softmax"]

    # Store relevant parameters in a dictionary.
    params = {
        "data_name": data_name,
        "guide_model_name_or_path": guide_model_name_or_path,
        # "cache_folder": cache_folder,
        # "save_dir": save_dir,
        "lang": lang,
        "topk": topk,
        "temperature": temperature,
        # "device": device,
        "seed": seed,
        "negative_sampling": negative_sampling,
        "neg_topk": neg_topk,
    }

    payload = dict(
        path=data_name,
        revision=revision,
        split="train",
    )

    data = datasets.load_dataset(
        **payload
    )

    params["revision"] = revision

    # First we take positive examples from texts in the same class.
    # Then, we mine negative examples from texts in different classes.
    # We use a guide model to mine negative examples.

    # We shuffle the data within the same class to avoid bias.
    # Then convert the text to embeddings.

    set_seed(seed)
    data = data.shuffle()

    # Add the index to the data.
    data = data.map(lambda example, idx: {"idx": idx, **example}, with_indices=True)

    if "skip_flag" in data.column_names:
        data = data.filter(lambda example: not example["skip_flag"])

    model = EncoderSentenceTransformer(guide_model_name_or_path, cache_folder=cache_folder, device=device)
    embeddings = model.encode(data["pos"], batch_size=encode_batch_size, show_progress_bar=True, convert_to_tensor=True)

    similarity_fct = nn.CosineSimilarity(dim=-1)

    sims = batch_sim(similarity_fct, embeddings if not use_cpu else embeddings.cpu(), sim_batch_size)

    # Set the diagonal of sims to 0.
    # This is to avoid the same text being selected as a negative example.
    sims = sims.fill_diagonal_(-torch.inf)

    # For each class, we select the positive and negative sample probabilistically.
    # Use use masks to select the positive and negative samples.

    cos_sim_pos = []
    cos_sim_neg = []
    query_idx = list(range(len(data)))
    positive_idx = []
    negative_idx = []

    # Mine the positive examples.
    pos_values, pos_indices = torch.topk(sims, topk)
    probs = nn.Softmax(dim=-1)(pos_values / temperature)
    pos_idx = torch.multinomial(probs, 1).squeeze(1)

    positive_idx.extend(pos_indices[torch.arange(len(sims)), pos_idx].tolist())
    cos_sim_pos.extend(pos_values[torch.arange(len(sims)), pos_idx].tolist())

    # Mine the negative examples.

    sims.fill_diagonal_(torch.inf)

    # Get the topk least similar examples (we add a negative sign to the similarity).
    # values, indices = torch.topk(-other_class_sims, min(topk, other_class_sims.size(1)))
    if neg_topk is None:
        assert sims.size(1) > (topk + 1), "The number of available examples considered for negative mining should be greater than topk + 1."
        neg_values, neg_indices = torch.topk(-sims, sims.size(1) - topk - 1)
    else:
        neg_values, neg_indices = torch.topk(-sims, min(neg_topk, sims.size(1)))

    if negative_sampling == "softmax":
        # We maintain the negative sign in the values to ensure that the least similar
        # examples are selected with the highest probability.
        probs = nn.Softmax(dim=-1)(neg_values / temperature)
    elif negative_sampling == "uniform":
        # I think we should not bias the selection of negative examples.
        # We should just select the negative examples randomly from the topk.
        probs = torch.ones_like(neg_values) / neg_values.size(1)
    else:
        raise ValueError(f"Unknown negative sampling: {negative_sampling}")

    neg_idx = torch.multinomial(probs, 1).squeeze(1)
    negative_idx = neg_indices[torch.arange(len(sims)), neg_idx].tolist()
    cos_sim_neg = (-neg_values[torch.arange(len(sims)), neg_idx]).tolist()

    # Sort the data by the query_idx.
    query_idx, positive_idx, negative_idx = zip(*sorted(zip(query_idx, positive_idx, negative_idx), key=lambda x: x[0]))

    data = data.add_column("query_idx", list(query_idx))
    data = data.add_column("positive_idx", list(positive_idx))
    data = data.add_column("negative_idx", list(negative_idx))
    data = data.add_column("cos_sim_pos", cos_sim_pos)
    data = data.add_column("cos_sim_neg", cos_sim_neg)

    # Use "-" instead of "_" since the convention for MTEB is to use "_" for dataset name.
    if data_name.startswith("avsolatorio/"):
        data_name = data_name.replace("avsolatorio/", "")

    data_name = data_name.replace("/", "-")

    try:
        data.save_to_disk(os.path.join(save_dir, f"{data_name}-avs_triplets"))
        data.push_to_hub(f"avsolatorio/{data_name}-avs_triplets", private=True, commit_message=json.dumps(params))
    except Exception as e:
        print(e)

    if return_data:
        return data


if __name__ == "__main__":
    """
poetry run python covid/scripts/create_covid_triplets_dataset.py \
    --data_name="avsolatorio/covid-bing-query-gpt4" \
    --guide_model_name_or_path="WhereIsAI/UAE-Large-V1" \
    --cache_folder="./cache_dir/" \
    --save_dir="mteb/avsolatorio" \
    --revision="f648ca46498605146c5757260862324471165506" \
    --device="mps" \
    --topk=10 \
    --neg_topk=None \
    --encode_batch_size=64 \
    --sim_batch_size=8
    """
    fire.Fire(mine_COVID)


"""
Classification MTEB datasets
# - mteb/amazon_counterfactual
- mteb/amazon_polarity
- mteb/amazon_reviews_multi
# - mteb/banking77
# - mteb/emotion
# - mteb/imdb
# - mteb/amazon_massive_intent
# - mteb/amazon_massive_scenario
# - mteb/mtop_domain
# - mteb/mtop_intent
- mteb/toxic_conversations_50k
# - mteb/tweet_sentiment_extraction
"""
import json
import os
import torch
import torch.nn as nn
import random
from mteb.tasks import Classification
from gist_embed.base import EncoderSentenceTransformer
from tqdm.auto import tqdm
import datasets
# Import set_seed
from transformers import set_seed


def batch_sim(similarity_fct, embeddings, batch_size):
    sq1 = embeddings.unsqueeze(1)
    sims_list = []

    for start_index in tqdm(range(0, len(embeddings), batch_size)):
        sims_list.append(
            similarity_fct(sq1, embeddings[start_index:start_index+batch_size].unsqueeze(0))
        )

        try:
            torch.cuda.empty_cache()
        except:
            try:
                torch.mps.empty_cache()
            except:
                pass

    sims = torch.cat(sims_list, dim=1)

    return sims


def mine_Classification(data_name: str, guide_model_name_or_path: str, cache_folder: str, save_dir: str, lang: str = "en", topk: int = 100, temperature: float = 0.05, device: str = None, seed: int = 1029, negative_sampling: str = "uniform", neg_topk: int = None, use_cpu: bool = False, encode_batch_size: int = 64):
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
    languages = None

    if data_name.endswith("amazon_counterfactual"):
        class_cls = Classification.AmazonCounterfactualClassification
    elif data_name.endswith("amazon_polarity"):
        class_cls = Classification.AmazonPolarityClassification
    elif data_name.endswith("amazon_reviews_multi"):
        class_cls = Classification.AmazonReviewsClassification
        languages = [lang]
    elif data_name.endswith("banking77"):
        class_cls = Classification.Banking77Classification
    elif data_name.endswith("emotion"):
        class_cls = Classification.EmotionClassification
    elif data_name.endswith("imdb"):
        class_cls = Classification.ImdbClassification
    elif data_name.endswith("amazon_massive_intent"):
        class_cls = Classification.MassiveIntentClassification
    elif data_name.endswith("amazon_massive_scenario"):
        class_cls = Classification.MassiveScenarioClassification
    elif data_name.endswith("mtop_domain"):
        class_cls = Classification.MTOPDomainClassification
    elif data_name.endswith("mtop_intent"):
        class_cls = Classification.MTOPIntentClassification
    elif data_name.endswith("toxic_conversations_50k"):
        class_cls = Classification.ToxicConversationsClassification
    elif data_name.endswith("tweet_sentiment_extraction"):
        class_cls = Classification.TweetSentimentExtractionClassification
    else:
        raise ValueError(f"Unknown dataset: {data_name}")

    cl = class_cls()
    payload = dict(
        path=data_name,
        revision=cl.description.get("revision", None),
        split="train",
    )

    if languages:
        payload["languages"] = languages

    data = datasets.load_dataset(
        **payload
    )

    params["revision"] = cl.description.get("revision", None)

    # cl.load_data()
    # try:
    #     data = cl.dataset[lang]["train"]
    # except KeyError:
    #     data = cl.dataset["train"]

    # First we take positive examples from texts in the same class.
    # Then, we mine negative examples from texts in different classes.
    # We use a guide model to mine negative examples.

    # We shuffle the data within the same class to avoid bias.
    # Then convert the text to embeddings.

    set_seed(seed)
    data = data.shuffle()

    # Add the index to the data.
    data = data.map(lambda example, idx: {"idx": idx, **example}, with_indices=True)

    model = EncoderSentenceTransformer(guide_model_name_or_path, cache_folder=cache_folder, device=device)
    embeddings = model.encode(data["text"], batch_size=encode_batch_size, show_progress_bar=True, convert_to_tensor=True)

    similarity_fct = nn.CosineSimilarity(dim=-1)

    sims = batch_sim(similarity_fct, embeddings if not use_cpu else embeddings.cpu(), 32)

    # Set the diagonal of sims to 0.
    # This is to avoid the same text being selected as a negative example.
    sims = sims.fill_diagonal_(-torch.inf)

    # For each class, we select the positive and negative sample probabilistically.
    # Use use masks to select the positive and negative samples.

    query_idx = []
    positive_idx = []
    negative_idx = []

    for label in set(data["label"]):
        print("Processing label:", label)
        class_idx = data.filter(lambda example: example["label"] == label)["idx"]

        if len(class_idx) <= 1:
            query_idx.extend(class_idx)
            positive_idx.extend([None] * len(class_idx))
            negative_idx.extend([None] * len(class_idx))
            continue

        class_idx = torch.tensor(class_idx, dtype=torch.long, device=sims.device)

        # For positive examples
        class_sims = sims[class_idx.view(-1, 1), class_idx.view(1, -1)]
        class_sims.fill_diagonal_(-torch.inf)

        values, indices = torch.topk(class_sims, min(topk, class_sims.size(1)))

        probs = nn.Softmax(dim=-1)(values / temperature)

        pos_idx = torch.multinomial(probs, 1).squeeze(1)

        # These are the positive examples indices.
        query_idx.extend(class_idx.tolist())
        positive_idx.extend(class_idx[indices[torch.arange(len(indices)).long(), pos_idx]].tolist())

        # For negative examples
        # We select the negative examples from the other classes.
        other_class_idx = data.filter(lambda example: example["label"] != label)["idx"]
        other_class_idx = torch.tensor(other_class_idx, dtype=torch.long, device=sims.device)

        other_class_sims = sims[class_idx.view(-1, 1), other_class_idx.view(1, -1)]

        # Get the topk least similar examples (we add a negative sign to the similarity).
        # values, indices = torch.topk(-other_class_sims, min(topk, other_class_sims.size(1)))
        if neg_topk is None:
            values, indices = torch.topk(-other_class_sims, other_class_sims.size(1))
        else:
            values, indices = torch.topk(-other_class_sims, min(neg_topk, other_class_sims.size(1)))

        if negative_sampling == "softmax":
            # We maintain the negative sign in the values to ensure that the least similar
            # examples are selected with the highest probability.
            probs = nn.Softmax(dim=-1)(values / temperature)
        elif negative_sampling == "uniform":
            # I think we should not bias the selection of negative examples.
            # We should just select the negative examples randomly from the topk.
            probs = torch.ones_like(values) / values.size(1)
        else:
            raise ValueError(f"Unknown negative sampling: {negative_sampling}")

        neg_idx = torch.multinomial(probs, 1).squeeze(1)
        negative_idx.extend(other_class_idx[indices[torch.arange(len(indices)).long(), neg_idx]].tolist())

    # Sort the data by the query_idx.
    query_idx, positive_idx, negative_idx = zip(*sorted(zip(query_idx, positive_idx, negative_idx), key=lambda x: x[0]))

    data = data.add_column("query_idx", list(query_idx))
    data = data.add_column("positive_idx", list(positive_idx))
    data = data.add_column("negative_idx", list(negative_idx))

    # Use "-" instead of "_" since the convention for MTEB is to use "_" for dataset name.
    data_name = data_name.replace("/", "-")

    try:
        data.save_to_disk(os.path.join(save_dir, f"{data_name}-avs_triplets"))
        data.push_to_hub(f"avsolatorio/{data_name}-avs_triplets", private=True, commit_message=json.dumps(params))
    except Exception as e:
        print(e)

    return data


# data = mine_Classification("mteb/toxic_conversations_50k", "WhereIsAI/UAE-Large-V1", "./cache_dir/", "mteb/avsolatorio", device="cuda", neg_topk=None)
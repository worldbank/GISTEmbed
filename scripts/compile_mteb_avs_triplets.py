import os
import json
import datasets
from tqdm.auto import tqdm
import subprocess as sub

sub_out = sub.run(["git", "rev-parse",  "HEAD"], stdout=sub.PIPE)
git_repo_version = sub_out.stdout.strip().decode()


# Template
# query: ['', ''], pos: ['', ''], neg: ['', ''], task_name: ''

def fill_example(example):
    if example["positive_idx"] is None:
        example["positive_idx"] = example["query_idx"]

    if example["negative_idx"] is None:
        example["negative_idx"] = example["query_idx"]

    return example

metadata = {}

dataset_name_revision = {
    "avsolatorio/mteb-amazon_counterfactual-avs_triplets": "604d8278fec31acaa7e39f5cb0d36d7741664e62",
    "avsolatorio/mteb-amazon_reviews_multi-avs_triplets": "0466b8190879e9952cf18960327bc0237c4f2c85",
    "avsolatorio/mteb-amazon_massive_intent-avs_triplets": "2a4e9393a357e0e68b8436eb2d952bf70668247a",
    "avsolatorio/mteb-amazon_massive_scenario-avs_triplets": "703d30bb58aea60d7f2886c9a5ed145f6850f824",
    "avsolatorio/mteb-banking77-avs_triplets": "3d313d5bd58d89a52a584022a98b910afb8cafd6",
    "avsolatorio/mteb-emotion-avs_triplets": "8fbb38668b259d8d639adcb328aab96c32c97db6",
    "avsolatorio/mteb-imdb-avs_triplets": "25d977fc6505cc888562f6ba4c97a5b24f5dda6f",
    "avsolatorio/mteb-mtop_domain-avs_triplets": "68e8a6958879c8713678d25ea652e48a047a7a02",
    "avsolatorio/mteb-mtop_intent-avs_triplets": "44da9dec812d8a991bd4604b5e96238e9358f60e",
    "avsolatorio/mteb-toxic_conversations_50k-avs_triplets": "564f7f1ee58406ffd00dd4e4e5d2a63bf534f879",
    "avsolatorio/mteb-tweet_sentiment_extraction-avs_triplets": "a0d5a4d8090337fc91266ab26fe2ad9471bfdddf",
}

dataset_name_revision = {k: v for k, v in sorted(dataset_name_revision.items())}

# medi_name = "medi-data"
# medi_name = "medi-data-sorted"
# medi_name = "medi-data-sorted_WhereIsAI_UAE-Large-V1"

def load_mteb_data(dataset_name_revision):
    mteb_data = []
    for data_name, revision in tqdm(dataset_name_revision.items(), desc="Dataset"):
        task_name = data_name.split("/")[-1]

        data = datasets.load_dataset(
            data_name, split="train",
            revision=revision,
        )
        data = data.map(fill_example)

        for q, p, n in tqdm(zip(data[data["query_idx"]]["text"], data[data["positive_idx"]]["text"], data[data["negative_idx"]]["text"]), desc="Samples"):
            if q == p or q == n:
                continue

            mteb_data.append(
                dict(
                    query=["", q],
                    pos=["", p],
                    neg=["", n],
                    task_name=task_name,
                )
            )

    return mteb_data


def build_medi_mteb_dataset(medi_name, mteb_data=None):
    MEDI_DATA_PATH = f"medi-data/{medi_name}.json"

    if medi_name in ("medi-data", "medi-data-sorted"):
        MEDI_DATA = json.load(open("medi-data/medi-data.json", "r"))

        if medi_name == "medi-data-sorted":
            MEDI_DATA = sorted(MEDI_DATA, key=lambda o: o["task_name"])
            MEDI_DATA_PATH = f"medi-data/{medi_name}.json"
    else:
        MEDI_DATA = json.load(open(MEDI_DATA_PATH, "r"))
        num_samples = len(MEDI_DATA)

    if medi_name == "medi-data":
        metadata["base_data"] = {"source": MEDI_DATA_PATH,  "description": "Original MEDI."}
    elif medi_name == "medi-data-sorted":
        metadata["base_data"] = {"source": MEDI_DATA_PATH,  "description": "Original MEDI sorted by `task_name`."}
    elif medi_name == "medi-data-sorted_WhereIsAI_UAE-Large-V1":
        metadata["base_data"] = {"source": MEDI_DATA_PATH,  "description": "Used the model WhereIsAI/UAE-Large-V1 with mean pooling to validate the MEDI data qp > qn."}
    else:
        raise ValueError(f"Invalid medi_name: {medi_name}")

    metadata["git_repo_version"] = git_repo_version
    metadata["dataset_name_revision"] = dataset_name_revision

    dataset = dict(metadata=metadata)
    dataset["data"] = MEDI_DATA

    if mteb_data is None:
        mteb_data = load_mteb_data(dataset_name_revision)

    dataset["data"].extend(mteb_data)

    query = []
    pos = []
    neg = []
    query_instruct = []
    pos_instruct = []
    neg_instruct = []
    task_name = []

    for o in dataset["data"]:
        query.append(o["query"][1])
        pos.append(o["pos"][1])
        neg.append(o["neg"][1])

        query_instruct.append(o["query"][0])
        pos_instruct.append(o["pos"][0])
        neg_instruct.append(o["neg"][0])

        task_name.append(o["task_name"])


    dd = datasets.Dataset.from_dict(
        mapping=dict(
            query=query,
            pos=pos,
            neg=neg,
            task_name=task_name,
            query_instruct=query_instruct,
            pos_instruct=pos_instruct,
            neg_instruct=neg_instruct,
        ),
        info=datasets.DatasetInfo(description=json.dumps(dataset["metadata"])),
        split="train",
    )

    dd.push_to_hub(
        f"avsolatorio/{medi_name}-mteb_avs_triplets",
        private=True,
        commit_description=json.dumps(dataset["metadata"])
    )

    with open(os.path.join("mteb/avsolatorio", f"{medi_name}-mteb_avs_triplets.json"), "w") as fl:
        json.dump(dataset, fl)

    if medi_name == "medi-data-sorted_WhereIsAI_UAE-Large-V1":
        if "dataset_name_revision" in dataset["metadata"]:
            dataset["metadata"].pop("dataset_name_revision")

        dd = datasets.Dataset.from_dict(
            mapping=dd[:num_samples],
            info=datasets.DatasetInfo(description=json.dumps(dataset["metadata"])),
            split="train",
        )

        dd.push_to_hub(
            f"avsolatorio/{medi_name}",
            private=True,
            commit_description=json.dumps(dataset["metadata"])
        )


def main():
    mteb_data = load_mteb_data(dataset_name_revision)

    for medi_name in ["medi-data", "medi-data-sorted", "medi-data-sorted_WhereIsAI_UAE-Large-V1"]:
        print(medi_name)
        build_medi_mteb_dataset(medi_name, mteb_data=mteb_data)


if __name__ == "__main__":
    main()
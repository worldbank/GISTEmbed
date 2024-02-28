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


# medi_name = "medi-data"
# medi_name = "medi-data-sorted"
# medi_name = "medi-data-sorted_WhereIsAI_UAE-Large-V1"

def load_covid_triplet_data(dataset_name_revision):
    dataset = []

    for data_name, revision in tqdm(dataset_name_revision.items(), desc="Dataset"):
        task_name = data_name.split("/")[-1]

        data = datasets.load_dataset(
            data_name, split="train",
            revision=revision,
        )
        data = data.map(fill_example)

        # Get the negative examples from the `negative_idx`.

        for q, p, n in tqdm(zip(data["query"], data["pos"], data[data["negative_idx"]]["pos"]), desc="Samples"):
            if q == p or q == n:
                continue

            dataset.append(
                dict(
                    query=["", q],
                    pos=["", p],
                    neg=["", n],
                    task_name=task_name,
                )
            )

    return dataset


def main():
    covid_data_name = "covid-bing-query-gpt4-avs_triplets"

    metadata = {
        "description": "MEDI + MTEB + COVID BING QUERY.",
        "git_repo_version": git_repo_version,
        "datasets": {
            "medi-data-mteb_avs_triplets": {
                "source": "avsolatorio/medi-data-mteb_avs_triplets",
                "revision": "238a0499b6e6b690cc64ea56fde8461daa8341bb"
            },
            covid_data_name: {
                "source": f"avsolatorio/{covid_data_name}",
                "revision": "0d2e2ff6c5f76a1ae74f7e545a03e1f4f36017b3"
            }
        }
    }

    dataset_name_revision = {
        metadata["datasets"][covid_data_name]["source"]: metadata["datasets"][covid_data_name]["revision"],
    }

    # Load the covid-bing-query-gpt4-avs_triplets
    covid_data = load_covid_triplet_data(dataset_name_revision)

    # Structure the data.
    query = []
    pos = []
    neg = []
    query_instruct = []
    pos_instruct = []
    neg_instruct = []
    task_name = []

    for o in covid_data:
        query.append(o["query"][1])
        pos.append(o["pos"][1])
        neg.append(o["neg"][1])

        query_instruct.append(o["query"][0])
        pos_instruct.append(o["pos"][0])
        neg_instruct.append(o["neg"][0])

        task_name.append(o["task_name"])

    covid_data = datasets.Dataset.from_dict(
        mapping=dict(
            query=query,
            pos=pos,
            neg=neg,
            task_name=task_name,
            query_instruct=query_instruct,
            pos_instruct=pos_instruct,
            neg_instruct=neg_instruct,
        ),
        split="train",
    )

    # Load medi-data-mteb_avs_triplets
    data_name = metadata["datasets"]["medi-data-mteb_avs_triplets"]["source"]
    revision = metadata["datasets"]["medi-data-mteb_avs_triplets"]["revision"]

    data = datasets.load_dataset(
        data_name, split="train",
        revision=revision,
    )

    final = datasets.concatenate_datasets([data, covid_data])

    final.push_to_hub(
        f"avsolatorio/medi-data-mteb-{covid_data_name}",
        private=True,
        commit_description=json.dumps(metadata),
        split="train",
    )


if __name__ == "__main__":
    main()
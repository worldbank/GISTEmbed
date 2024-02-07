import json
import pandas as pd
import numpy as np
from huggingface_hub import HfApi
import fire


def infer_related_repos(repo_id):
    # This should handle the case when we had to manually split a model into multiple repos.
    repo_ids = [repo_id]
    cand_repo = repo_id.replace("0-u_GIST", "0_GIST")
    if cand_repo != repo_id:
        repo_ids.append(cand_repo)

    return repo_ids


def get_best_checkpoint(repo_ids, min_step, conf=0.025, pos=None, max_step=100_000):
    # We use repo_ids since some models have been split into multiple repos.
    api = HfApi()

    data = []

    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]

    for repo_id in repo_ids:
        commits = api.list_repo_commits(repo_id)

        for commit in commits:
            try:
                data.append(json.loads(commit.title))
                data[-1]["commit_id"] = commit.commit_id
                data[-1]["repo_id"] = repo_id
            except:
                print("Failed to load commit:", commit.title)

    data = pd.DataFrame(data)
    data.drop_duplicates(subset=["step", "loss"], inplace=True)

    data = data[(data["step"] >= min_step) & (data["step"] <= max_step)]

    # Do this assertion since we ran the training for 100k steps with
    # checkpoints every 500 steps.
    assert data["step"].max() == max_step and data["step"].min() == min_step

    data = data.sort_values("loss", ascending=True)

    if pos is None:
        pos = int(len(data) * conf)
    else:
        print("Using pos:", pos, "instead of", int(len(data) * conf))

    # Get the checkpoint corresponding to the confidence interval.
    best = dict(data.iloc[pos])

    return best


def main(repo_id, min_step, conf=0.025, pos=None, max_step=100_000):
    repo_ids = infer_related_repos(repo_id)
    best = get_best_checkpoint(repo_ids, conf=conf, min_step=min_step, pos=pos, max_step=max_step)

    print(f"Best checkpoint: {best}")

    return best


if __name__ == "__main__":
    # poetry run python scripts/parse_commits.py --repo_id=avsolatorio/00-100-11-1-0-1-0-0-cls_GIST_BAAI_bge-base-en-v1.5-20240127003236-latest --conf=0.025

    fire.Fire(main)

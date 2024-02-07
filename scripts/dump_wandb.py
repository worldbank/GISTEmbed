import os
import json
import wandb

import fire


def dump_data(run_id: str, out_dir: str):
    # run_id = "/avsolatorio/GIST_BAAI_bge-base-en-v1.5/runs/iqg3llwm"

    wandb_dir = os.path.join(out_dir, "wandb")
    os.makedirs(wandb_dir, exist_ok=True)

    api = wandb.Api()
    run = api.run(run_id)

    # Dump the metadata.
    json.dump(run.metadata, open(os.path.join(wandb_dir, "metadata.json"), "w"), indent=2)

    # Dump the history
    run.history().to_json(os.path.join(wandb_dir, "history.json"), orient="records", indent=2)

    # Dump json_config
    json.dump(run.json_config, open(os.path.join(wandb_dir, "json_config.json"), "w"), indent=2)


if __name__ == "__main__":
    # poetry run python scripts/dump_wandb.py --run_id=/avsolatorio/GIST_BAAI_bge-base-en-v1.5/runs/iqg3llwm --out_dir=published/model/00-100-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240129004245-latest
    fire.Fire(dump_data)


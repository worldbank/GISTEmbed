import os
import json
from tqdm.auto import tqdm
from hashlib import md5
from fire import Fire
from collections import defaultdict

def hash_item(item):
    query = "".join([i.strip().rstrip(":").rstrip(";").rstrip(":") for i in item["query"]]).lower().replace(" ", "")
    pos = "".join([i.strip() for i in item["pos"]]).lower().replace(" ", "")
    neg = "".join([i.strip() for i in item["neg"]]).lower().replace(" ", "")
    item = query + pos + neg

    return md5(item.encode('utf-8')).hexdigest()


def create_item_hash2task(item):
    if "task_name" in item:
        task = item.pop("task_name")
    elif "task_id" in item:
        task = item.pop("task_id")
    else:
        raise ValueError("No task_name or task_id in item")

    return {hash_item(item): task}


def hash_query(query):
    query = query[0].lower().replace(" ", "").rstrip(":").rstrip(";").rstrip(":").rstrip(";")
    return md5(query.encode('utf-8')).hexdigest()


def run(base_dir, out_dir):
    # Load the medi-data with task_name

    with open(os.path.join(base_dir, "medi-data_task-name", 'medi-data.json')) as f:
        task_name_data  = json.load(f)


    # Load the medi-data with task_id
    with open(os.path.join(base_dir, "medi-data_task-id", 'medi-data.json')) as f:
        task_id_data  = json.load(f)

    instruction2tasks = defaultdict(dict)

    task_name2instruction = defaultdict(set)
    task_id2instruction = defaultdict(set)

    hash2name = {}

    for item in tqdm(task_name_data, desc="Creating hash2name"):
        task_name2instruction[item["task_name"]].add(item["query"][0])
        qhash = hash_query(item["query"])
        if "task_name" not in instruction2tasks[qhash]:
            instruction2tasks[qhash]["task_name"] = []

        if "instruction" not in instruction2tasks[qhash]:
            instruction2tasks[qhash]["instruction"] = []

        if item["task_name"] not in instruction2tasks[qhash]["task_name"]:
            instruction2tasks[qhash]["task_name"].append(item["task_name"])
            instruction2tasks[qhash]["instruction"].append(item["query"][0])


        hash2name.update(create_item_hash2task(item))

    hash2id = {}

    for item in tqdm(task_id_data, desc="Creating hash2id"):
        task_id2instruction[item["task_id"]].add(item["query"][0])
        qhash = hash_query(item["query"])

        if "task_id" not in instruction2tasks[qhash]:
            instruction2tasks[qhash]["task_id"] = []

        if "instruction" not in instruction2tasks[qhash]:
            instruction2tasks[qhash]["instruction"] = []

        if item["task_id"] not in instruction2tasks[qhash]["task_id"]:
            instruction2tasks[qhash]["task_id"].append(item["task_id"])
            instruction2tasks[qhash]["instruction"].append(item["query"][0])

        hash2id.update(create_item_hash2task(item))

    medi_task_name2id = {}

    for item_hash in tqdm(hash2name, desc="Creating medi_task_name2id"):
        if item_hash in hash2id:
            medi_task_name2id[hash2name[item_hash]] = hash2id[item_hash]

    unique_task_names = set(hash2name.values())
    unique_task_ids = set(hash2id.values())

    json.dump(medi_task_name2id, open(os.path.join(out_dir, "medi-data_task-name2id.json"), "w"), indent=2)

    print(f"Unique task names: {len(unique_task_names)}")
    print(f"Unique task ids: {len(unique_task_ids)}")

    json.dump(list(unique_task_names), open(os.path.join(out_dir, "medi-data_task-names.json"), "w"), indent=2)
    json.dump(list(unique_task_ids), open(os.path.join(out_dir, "medi-data_task-ids.json"), "w"), indent=2)

    task_name2instruction = {k: list(v) for k, v in task_name2instruction.items()}
    task_id2instruction = {k: list(v) for k, v in task_id2instruction.items()}

    json.dump(dict(task_name2instruction), open(os.path.join(out_dir, "medi-data_task-name2instruction.json"), "w"), indent=2)
    json.dump(dict(task_id2instruction), open(os.path.join(out_dir, "medi-data_task-id2instruction.json"), "w"), indent=2)

    json.dump(dict(instruction2tasks), open(os.path.join(out_dir, "medi-data_instruction2tasks.json"), "w"), indent=2)

if __name__ == "__main__":
    # poetry run python scripts/medi_data_task_name2id.py  --base_dir /Users/avsolatorio/Downloads --out_dir ./
    Fire(run)

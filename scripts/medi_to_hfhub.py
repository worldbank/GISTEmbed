import json
import datasets
from tqdm.auto import tqdm


commit_description = """
# This is comes from the MEDI data published with the paper "One Embedder, Any Task: Instruction-Finetuned Text Embeddings" [https://github.com/xlang-ai/instructor-embedding]

## Download the data raw data

```Bash
pip install gdown
gdown --id 1vZ5c2oJNonGOvXzppNg5mHz24O6jcc52 -O medi-data.zip
unzip medi-data.zip
```
"""

MEDI_DATA = "medi-data/medi-data.json"
medi_data = json.load(open(MEDI_DATA, "r"))

query = []
pos = []
neg = []
query_instruct = []
pos_instruct = []
neg_instruct = []
task_name = []

for o in medi_data:
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
    split="train",
)

dd.push_to_hub(
    "avsolatorio/medi-data",
    private=True,
    commit_description=commit_description
)

sorted_dd = dd.sort(["task_name"])


commit_description += "\n\n## Note\n\nThis version is sorted by `task_name`."

sorted_dd.push_to_hub(
    "avsolatorio/medi-data-sorted",
    private=True,
    commit_description=commit_description
)

# with open(os.path.join("mteb/avsolatorio", "medi-data_WhereIsAI_UAE-Large-V1-mteb_avs_triplets.json"), "w") as fl:
#     json.dump(dataset, fl)

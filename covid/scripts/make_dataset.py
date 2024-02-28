import os
import glob
import json
import pandas as pd
import datasets

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()


def is_valid_triplet(triplet):
    if not isinstance(triplet, dict):
        return False

    if "s" not in triplet:
        return False

    if "q" not in triplet:
        return False

    if "p" not in triplet:
        return False

    return True

def parse_file(file):
    with open(file, "r") as f:
        m = json.load(f)
        o = parser.parse(m["choices"][0]["message"]["content"])
        if o is None:
            print(f"Failed to parse {file}")
            return []
        else:
            return o

# Get the parent directory of the directory where this script is located
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# Get the list of all the triplet files in the data folder
files = sorted(glob.glob(os.path.join(parent_dir, "data", "triplets", "*.json")))

# Read the data from all the files and concatenate them into a single list
triplets = []

for f in files:
    print(f"Parsing {f}")
    triplets.extend(parse_file(f))

print(len(triplets))
triplets = [t for t in triplets if is_valid_triplet(t)]
print(len(triplets))

df = pd.DataFrame.from_records(triplets)
df.rename(columns={"s": "search", "q": "query", "p": "pos"}, inplace=True)

data = datasets.Dataset.from_pandas(df)
data_name = "avsolatorio/covid-bing-query-gpt4"
data.push_to_hub(data_name, private=True, commit_message=f"Add dataset.")

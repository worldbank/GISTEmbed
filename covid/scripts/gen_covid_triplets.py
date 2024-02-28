import os
import json

from tqdm.auto import tqdm
from openai import OpenAI
client = OpenAI()


# Get the directory where this script is located
dir_path = os.path.dirname(os.path.realpath(__file__))

# Get the parent directory of the directory where this script is located
parent_dir = os.path.dirname(dir_path)


template = lambda terms: """Generate a dataset of training triplets of COVID-19 queries, each with a positive example or relevant passage. The queries must be based on the search terms provided. Return the data as JSON records [{"s": <search terms>, "q": <query>, "p": <positive>}...]

search terms:\n- """ + "\n- ".join(terms) + "\n\n```json"

batch = 20
queries = json.load(open(os.path.join(parent_dir, "data/queries/covid_top_queries.json")))


for i in tqdm(range(0, len(queries), batch)):
    payload = dict(
        role="system",
        content=template([q["Query"] for q in queries[i:i+batch]])
    )

    with open(os.path.join(parent_dir, f"data/payloads/message{i + batch:05}.json"), "w") as f:
        json.dump(payload, f, indent=2)

    triplet_file = os.path.join(parent_dir, f"data/triplets/triplets{i + batch:05}.json")

    # Skip if the file already exists
    if os.path.exists(triplet_file):
        continue

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[payload],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    with open(triplet_file, "w") as f:
        json.dump(response.model_dump(), f, indent=2)

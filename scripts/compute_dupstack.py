import os
import json
import fire


CQADupstack = [
    "CQADupstackWebmastersRetrieval",  # 135.36 sec
    "CQADupstackAndroidRetrieval",  # 141.32 sec
    "CQADupstackEnglishRetrieval",  # 212.31 sec
    "CQADupstackMathematicaRetrieval",  # 212.78 sec
    "CQADupstackGamingRetrieval",  # 231.24 sec
    "CQADupstackProgrammersRetrieval",  # 301.38 sec
    "CQADupstackGisRetrieval",  # 356.43 sec
    "CQADupstackPhysicsRetrieval",  # 384.95 sec
    "CQADupstackUnixRetrieval",  # 439.81 sec
    "CQADupstackStatsRetrieval",  # 487.38 sec
    "CQADupstackWordpressRetrieval",  # 489.85 sec
    "CQADupstackTexRetrieval",  # 813.53 sec
]


def get_dup_score(path):
    retrieval = os.path.join(path, "Retrieval")
    assert os.path.exists(retrieval)

    scores = []

    for cqd in CQADupstack:
        cqd_path = os.path.join(retrieval, f"{cqd}.json")

        if not os.path.exists(cqd_path):
            raise ValueError(f"Missing CQADupstack: {cqd_path}")

        with open(cqd_path) as f:
            data = json.load(f)

        scores.append(data["test"]["ndcg_at_10"])

    return scores, sum(scores) / len(scores)


def main(path):
    scores, avg = get_dup_score(path)
    print(f"Scores: {scores}")
    print(f"Average: {avg}")


if __name__ == "__main__":
    fire.Fire(main)

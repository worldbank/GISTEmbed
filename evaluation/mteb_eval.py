import os
import json
import logging
import fire
from mteb import MTEB

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("main")

# model_id = "avsolatorio/MixEmbed-base-20240111100024-best"; model = gist_embed.MixSentenceTransformer(model_id, cache_folder="./cache_dir", device="cuda:0")
# run_mteb(model, model_id, TASK_LIST_STS, compute_score=True); run_mteb(model, model_id, TASK_LIST_PAIR_CLASSIFICATION, compute_score=True); run_mteb(model, model_id, TASK_LIST_RERANKING, compute_score=True); run_mteb(model, model_id, TASK_LIST_RETRIEVAL, compute_score=True); run_mteb(model, model_id, TASK_LIST_CLASSIFICATION, compute_score=True); run_mteb(model, model_id, TASK_LIST_CLUSTERING, compute_score=True)


NORMED_TASKS = [
    "AmazonReviewsClassification",
    "EmotionClassification",
    "ToxicConversationsClassification",
    "TweetSentimentExtractionClassification",
    "ImdbClassification",
]

TASK_LIST_CLASSIFICATION = [
    'TweetSentimentExtractionClassification',
    'EmotionClassification',
    'MassiveScenarioClassification',
    'MTOPDomainClassification',
    'AmazonReviewsClassification',
    'MassiveIntentClassification',
    'AmazonCounterfactualClassification',
    'Banking77Classification',
    'MTOPIntentClassification',
    'ToxicConversationsClassification',
    'ImdbClassification',
    'AmazonPolarityClassification'
]

# [('TwentyNewsgroupsClustering', 15.01),
#  ('MedrxivClusteringS2S', 35.65),
#  ('BiorxivClusteringS2S', 42.4),
#  ('StackExchangeClustering', 206.77),
#  ('MedrxivClusteringP2P', 297.35),
#  ('RedditClustering', 344.87),
#  ('ArxivClusteringS2S', 372.12),
#  ('StackExchangeClusteringP2P', 379.25),
#  ('BiorxivClusteringP2P', 546.15),
#  ('RedditClusteringP2P', 1744.22),
#  ('ArxivClusteringP2P', 4344.96)]

# TASK_LIST_CLUSTERING = [
#     "TwentyNewsgroupsClustering",
#     "ArxivClusteringP2P",
#     "ArxivClusteringS2S",
#     "BiorxivClusteringP2P",
#     "BiorxivClusteringS2S",
#     "MedrxivClusteringP2P",
#     "MedrxivClusteringS2S",
#     "RedditClustering",
#     "RedditClusteringP2P",
#     "StackExchangeClustering",
#     "StackExchangeClusteringP2P",
# ]

TASK_LIST_CLUSTERING = [
    "ArxivClusteringP2P",  # 4344.96
    "RedditClusteringP2P",  # 1744.22
    "BiorxivClusteringP2P",  # 546.15
    "StackExchangeClusteringP2P",  # 379.25
    "ArxivClusteringS2S",  # 372.12
    "RedditClustering",  # 344.87
    "MedrxivClusteringP2P",  # 297.35
    "StackExchangeClustering",  # 206.77
    "BiorxivClusteringS2S",  # 42.4
    "MedrxivClusteringS2S",  # 35.65
    "TwentyNewsgroupsClustering",  # 15.01
]

TASK_LIST_PAIR_CLASSIFICATION = [
    "SprintDuplicateQuestions",
    "TwitterSemEval2015",
    "TwitterURLCorpus",
]

TASK_LIST_RERANKING = [
    'AskUbuntuDupQuestions',
    'StackOverflowDupQuestions',
    'SciDocsRR',
    'MindSmallReranking'
]

TASK_LIST_RETRIEVAL = [
    'ArguAna',
    'CQADupstackAndroidRetrieval',
    'CQADupstackWebmastersRetrieval',
    'CQADupstackEnglishRetrieval',
    'CQADupstackMathematicaRetrieval',
    'CQADupstackGamingRetrieval',
    'CQADupstackProgrammersRetrieval',
    'CQADupstackGisRetrieval',
    'CQADupstackPhysicsRetrieval',
    'CQADupstackUnixRetrieval',
    'CQADupstackWordpressRetrieval',
    'FiQA2018',
    'CQADupstackStatsRetrieval',
    'CQADupstackTexRetrieval',
    'DBPedia',
    'ClimateFEVER',
    'FEVER',
]

TASK_LIST_RETRIEVAL_NEW = [
    #
    "HotpotQA",
    "MSMARCO",
    "NQ",
    "NFCorpus",
    "QuoraRetrieval",
    "SCIDOCS",
    "SciFact",
    "Touche2020",
    "TRECCOVID",
]

TASK_LIST_RETRIEVAL += TASK_LIST_RETRIEVAL_NEW

TASK_LIST_STS = [
    'STS17',
    'BIOSSES',
    'STS16',
    'STSBenchmark',
    'STS13',
    'STS15',
    'STS12',
    'STS14',
    'STS22',
    'SICK-R',
    # "SummEval",
]

TASK_LIST_PRIORITY = TASK_LIST_STS + TASK_LIST_PAIR_CLASSIFICATION + TASK_LIST_CLASSIFICATION + TASK_LIST_RERANKING

TASK_LIST_SUMMARIZATION = [
    'SummEval',
]

TASK_LIST_BITEXT_MINING = [
    'BUCC',
    'Bornholmsk',
    'Tatoeba',
]

TASK_LIST = (
    # TASK_LIST_STS +
    # TASK_LIST_CLASSIFICATION +
    # TASK_LIST_CLUSTERING +
    # TASK_LIST_PAIR_CLASSIFICATION +
    # TASK_LIST_RERANKING +
    # TASK_LIST_RETRIEVAL
    # + TASK_LIST_STS
)

TASK_LIST = (
    TASK_LIST_STS +
    TASK_LIST_PAIR_CLASSIFICATION +
    TASK_LIST_RERANKING +
    TASK_LIST_RETRIEVAL +
    TASK_LIST_CLASSIFICATION +
    TASK_LIST_SUMMARIZATION +
    TASK_LIST_BITEXT_MINING +
    TASK_LIST_CLUSTERING
)

TASK_LIST_GROUP = {
    "STS": TASK_LIST_STS,
    "Classification": TASK_LIST_CLASSIFICATION,
    "Clustering": TASK_LIST_CLUSTERING,
    "PairClassification": TASK_LIST_PAIR_CLASSIFICATION,
    "Reranking": TASK_LIST_RERANKING,
    "Retrieval": TASK_LIST_RETRIEVAL,
    "Summarization": TASK_LIST_SUMMARIZATION,
    "BitextMining": TASK_LIST_BITEXT_MINING,
}

# ["STS", "PairClassification", "Reranking", "Retrieval", "Clustering", "Classification", "BitextMining", "Summarization"]
{'BitextMining',
 'Classification',
 'Clustering',
 'PairClassification',
 'Reranking',
 'Retrieval',
 'STS'}


TASK_NAMES_EVAL_TIME = [
    "BIOSSES",  # 0.57 sec
    "STS17",  # 0.79 sec
    "STS16",  # 3.25 sec
    "STSBenchmark",  # 3.62 sec
    "STS13",  # 3.89 sec
    "STS22",  # 6.17 sec
    "SummEval",  # 7.53 sec
    "AmazonCounterfactualClassification",  # 7.61 sec
    "STS15",  # 7.72 sec
    "STS12",  # 8.45 sec
    "MassiveScenarioClassification",  # 9.13 sec
    "MTOPDomainClassification",  # 9.14 sec
    "STS14",  # 9.41 sec
    "EmotionClassification",  # 9.64 sec
    "AskUbuntuDupQuestions",  # 10.44 sec
    "MassiveIntentClassification",  # 13.63 sec
    "SprintDuplicateQuestions",  # 14.63 sec
    "Banking77Classification",  # 18.38 sec
    "MTOPIntentClassification",  # 20.61 sec
    "TwitterSemEval2015",  # 21.85 sec
    "AmazonReviewsClassification",  # 22.54 sec
    "SICK-R",  # 24.33 sec
    "TwentyNewsgroupsClustering",  # 56.89 sec
    "TwitterURLCorpus",  # 67.17 sec
    "TweetSentimentExtractionClassification",  # 72.53 sec
    "ArguAna",  # 86.26 sec
    "StackOverflowDupQuestions",  # 120.02 sec
    # "CQADupstackWebmastersRetrieval",  # 135.36 sec
    # "CQADupstackAndroidRetrieval",  # 141.32 sec
    "SciDocsRR",  # 167.29 sec
    # "Tatoeba",  # 178.75 sec
    "ToxicConversationsClassification",  # 189.36 sec
    # "CQADupstackEnglishRetrieval",  # 212.31 sec
    # "CQADupstackMathematicaRetrieval",  # 212.78 sec
    # "CQADupstackGamingRetrieval",  # 231.24 sec
    "ImdbClassification",  # 297.75 sec
    # "CQADupstackProgrammersRetrieval",  # 301.38 sec
    # "CQADupstackGisRetrieval",  # 356.43 sec
    # "CQADupstackPhysicsRetrieval",  # 384.95 sec
    # "CQADupstackUnixRetrieval",  # 439.81 sec
    "FiQA2018",  # 447.38 sec
    # "CQADupstackStatsRetrieval",  # 487.38 sec
    # "CQADupstackWordpressRetrieval",  # 489.85 sec
    # "CQADupstackTexRetrieval",  # 813.53 sec
    "AmazonPolarityClassification",  # 1913.57 sec
]

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

EXT = [
    "MindSmallReranking",  # 6577.0 sec
    "DBPedia",  # 16142.86 sec
    "ClimateFEVER",  # 28153.44 sec
    "FEVER",  # 28645.17 sec
    # "BUCC",  # 2927.37 sec
]

TASK_NAMES_EVAL_TIME.extend(CQADupstack)
TASK_NAMES_EVAL_TIME.extend(EXT)
TASK_NAMES_EVAL_TIME.extend(TASK_LIST_RETRIEVAL_NEW)
TASK_NAMES_EVAL_TIME.extend(TASK_LIST_CLUSTERING)

TASK_NAMES_EVAL_TIME.append("TRECCOVID")

TASK_NAME_GROUP = {}

for task_group, task_list in TASK_LIST_GROUP.items():
    for task in task_list:
        TASK_NAME_GROUP[task] = task_group


def run_task_types(model, output_folder):
    # "results/avsolatorio/MixEmbed-base-20240109150023-best-33500"
    task_types = ["STS", "PairClassification", "Reranking", "Retrieval", "Clustering", "Classification", "BitextMining", "Summarization"]
    for task_type in task_types:
        evaluation = MTEB(task_langs=["en"], task_types=[task_type])
        evaluation.run(model, output_folder=output_folder, eval_splits=["test"])


def get_output_folder(args):
    model_id = args.model_name.replace("/","-")
    return os.path.join(args.output_dir, "results", model_id)


def run_eval(model, task, args):
    # logger.info(f"Running task: {task}")
    # eval_splits = ["dev"] if task == "MSMARCO" else ["test"]

    evaluation = MTEB(tasks=[task], task_langs=["en"])
    evaluation.run(
        model,
        output_folder=get_output_folder(args),
        eval_splits=[args.split], args=args,
    )


def save_scores(output_path, scores):
    scores = sorted(scores, key=lambda x: x["name"])

    with open(output_path, "w") as fl:
        json.dump(scores, fl)


def compute_sts(TASK_LIST_STS, output_folder=None, agg=False):
    # output_folder = get_output_folder(args)
    if output_folder is None:
        output_folder = "results/result"

    scores = []
    for task in TASK_LIST_STS:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        if "cos_sim" in o["test"]:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["cos_sim"]["spearman"]))
        elif "en-en" in o["test"]:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["en-en"]["cos_sim"]["spearman"]))
        elif "en" in o["test"]:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["en"]["cos_sim"]["spearman"]))
        else:
            print(o)
            raise ValueError

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_pair(TASK_LIST_PAIR_CLASSIFICATION, output_folder=None, agg=False):
    # output_folder = get_output_folder(args)
    if output_folder is None:
        output_folder = "results/result"

    scores = []
    for task in TASK_LIST_PAIR_CLASSIFICATION:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["cos_sim"]["ap"]))
        except Exception as e:
            print(o)
            raise e

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_reranking(TASK_LIST_RERANKING, output_folder: str = None, agg=False):
    if output_folder is None:
        output_folder = "results/result"

    scores = []

    for task in TASK_LIST_RERANKING:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["map"]))
        except Exception as e:
            print(o)
            raise e

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_retrieval(TASK_LIST_RETRIEVAL, output_folder=None, agg=False):
    if output_folder is None:
        output_folder = "results/result"

    CQADupstack_scores = []
    scores = []
    for task in TASK_LIST_RETRIEVAL:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            if o["mteb_dataset_name"].startswith("CQADupstack"):
                CQADupstack_scores.append(dict(name=o["mteb_dataset_name"], score=o["test"]["ndcg_at_10"]))
            else:
                if o["mteb_dataset_name"] == "MSMARCO":
                    scores.append(
                        dict(name=o["mteb_dataset_name"], score=o["dev"]["ndcg_at_10"]))
                else:
                    scores.append(
                        dict(name=o["mteb_dataset_name"], score=o["test"]["ndcg_at_10"]))
        except Exception as e:
            print(o)
            raise e

    scores.append(
        dict(
            name="CQADupstackRetrieval",
            score=sum(o["score"] for o in CQADupstack_scores) / len(CQADupstack_scores)
        )
    )

    save_scores(os.path.join(output_folder, "scores.json"), scores)
    save_scores(os.path.join(output_folder, "CQADupstack_scores.json"), CQADupstack_scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_clustering(TASK_LIST_CLUSTERING, output_folder=None, agg=False):
    if output_folder is None:
        output_folder = "results/result"

    scores = []
    for task in TASK_LIST_CLUSTERING:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["v_measure"]))
        except Exception as e:
            print(o)
            raise e

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_classification(TASK_LIST_CLASSIFICATION, output_folder=None, agg=False):
    if output_folder is None:
        output_folder = "results/result"

    scores = []
    for task in TASK_LIST_CLASSIFICATION:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            l = o["test"]

            if "en" in l:
                l = l["en"]

            score = l["accuracy"]

            scores.append(
                dict(name=o["mteb_dataset_name"], score=score))
        except Exception as e:
            print(o)
            raise e

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_summarization(TASK_LIST_SUMMARIZATION, output_folder=None, agg=False):
    if output_folder is None:
        output_folder = "results/result"

    scores = []
    for task in TASK_LIST_SUMMARIZATION:
        o = json.load(open(f"{output_folder}/{task}.json", "r"))

        try:
            scores.append(
                dict(name=o["mteb_dataset_name"], score=o["test"]["cos_sim"]["spearman"]))
        except Exception as e:
            print(o)
            raise e

    save_scores(os.path.join(output_folder, "scores.json"), scores)

    if agg:
        scores = 0 if len(scores) == 0 else sum([o["score"] for o in scores]) / len(scores)

    return scores


def compute_task_group_average(args):
    if args.task_name_group is None:
        return

    if args.task_name_group == "STS":
        scores = compute_sts(args)
        mean_score = sum([s["score"] for s in scores]) / len(scores)
    elif args.task_name_group == "Reranking":
        scores = compute_reranking(args)
        mean_score = sum([s["score"] for s in scores]) / len(scores)
    else:
        scores = []
        mean_score = 0

    return mean_score


class MTEBReport:
    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir

    def get_scores(self, task_name):
        pass


def run_mteb(model, model_id, tasks, cache_folder=None, output_folder=None, eval_splits=["test"], compute_score=False, batch_size=32, normed=None, use_normed_tasks=True, expected_step=None, rev_task=False):
    import torch
    from mteb import MTEB
    import json
    from sentence_transformers.models import Normalize

    if rev_task:
        tasks = tasks[::-1]

    orig_has_norm = model._last_module().__module__.endswith(".Normalize")
    concat_norm = getattr(model, "concat_norm", None)

    if normed is not None:
        if normed:
            if not orig_has_norm:
                model = model.append(Normalize())
        else:
            if orig_has_norm:
                model.pop(-1)

                assert not model._last_module().__module__.endswith(".Normalize")
                print(model)
    else:
        normed = orig_has_norm

    task_type_count = {
        k: len(v) for k, v in TASK_LIST_GROUP.items()
    }

    task_set = {
        'STS': (TASK_LIST_STS, compute_sts),
        'PairClassification': (TASK_LIST_PAIR_CLASSIFICATION, compute_pair),
        'Reranking': (TASK_LIST_RERANKING, compute_reranking),
        'Retrieval': (TASK_LIST_RETRIEVAL, compute_retrieval),
        'Classification': (TASK_LIST_CLASSIFICATION, compute_classification),
        'Summarization': (TASK_LIST_SUMMARIZATION, compute_summarization),
        # 'BitextMining': (TASK_LIST_BITEXT_MINING, compute_bitext_mining),
        'Clustering': (TASK_LIST_CLUSTERING, compute_clustering),
    }

    if cache_folder is None:
        cache_folder = "cache_dir"
    if output_folder is None:
        output_folder = "results"

    model_path = f"{cache_folder}/{model_id.replace('/', '_')}"
    steps = 0

    if hasattr(model, "commit_info_json_path") and model.commit_info_json_path is not None:
        commit_info_path = model.commit_info_json_path
    else:
        commit_info_path = f"{model_path}/commit-info.json"

    try:
        commit_info = json.load(open(commit_info_path, "r"))
    except FileNotFoundError:
        commit_info = None

    if commit_info is not None:
        try:
            commit_message = json.loads(commit_info["commit_message"])
            steps = commit_message["step"]
        except Exception as e:
            steps = commit_info["commit_message"].split()[4]

    m_id = model_id.split('/')[1]

    feature = "concat_norm" if concat_norm else ("normed" if normed else "unnormed")

    if use_normed_tasks:
        model.normed_tasks = NORMED_TASKS
        feature = "normed_tasks"

    if expected_step is not None:
        assert int(steps) == int(expected_step)

    results_dir = f"{output_folder}/{m_id}-{steps:06}-feature={feature}"

    if use_normed_tasks:
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "normed_tasks.json"), "w") as fl:
            json.dump(NORMED_TASKS, fl)

    for task in tasks:
        if use_normed_tasks:
            model.task = task
        task_type = TASK_NAME_GROUP[task]
        print(f"Running task: {task_type}::{task}")
        eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
        evaluation = MTEB(tasks=[task], task_langs=["en"])
        print(evaluation.run(
            model,
            output_folder=f"{results_dir}/{task_type}",
            eval_splits=eval_splits,
            batch_size=batch_size))
        torch.cuda.empty_cache()

        task_type_count[task_type] -= 1

        if task_type_count[task_type] == 0:
            # Compute the scores here
            scores = {}
            scores_path = os.path.join(results_dir, "scores.json")
            if os.path.exists(scores_path):
                scores = json.load(open(scores_path, "r"))

            _tasks, _score_func = task_set[task_type]

            try:
                scores[task_type] = _score_func(_tasks, output_folder=f"{results_dir}/{task_type}", agg=True)
                print(f"{task_type}: {scores[task_type]}, {scores_path} {scores}")

                with open(scores_path, "w") as fOut:
                    json.dump(scores, fOut, sort_keys=True, indent=2)

            except Exception as e:
                print(e)
                scores[task_type] = None

    # if compute_score:
    #     scores = {}

    #     task_set = [
    #         ('STS', TASK_LIST_STS, compute_sts),
    #         ('PairClassification', TASK_LIST_PAIR_CLASSIFICATION, compute_pair),
    #         ('Reranking', TASK_LIST_RERANKING, compute_reranking),
    #         ('Retrieval', TASK_LIST_RETRIEVAL, compute_retrieval),
    #         ('Classification', TASK_LIST_CLASSIFICATION, compute_classification),
    #         ('Summarization', TASK_LIST_SUMMARIZATION, compute_summarization),
    #         # ('BitextMining', TASK_LIST_BITEXT_MINING, compute_bitext_mining),
    #         # ('Clustering', TASK_LIST_CLUSTERING, compute_clustering),
    #     ]

    #     for task_type, tasks, score_func in task_set:
    #         try:
    #             scores[task_type] = score_func(tasks, output_folder=f"{results_dir}/{task_type}", agg=True)
    #             print(f"{task_type}: {scores[task_type]}")
    #         except Exception as e:
    #             print(e)
    #             scores[task_type] = None

    #     if scores:
    #         with open(f"{results_dir}/scores.json", "w") as fOut:
    #             json.dump(scores, fOut)

    return results_dir


def run_all_mteb(model, model_id, task_types=None, cache_folder=None, output_folder=None, eval_splits=["test"], compute_score=True, batch_size=32, normed=None, use_normed_tasks=False, expected_step=None):
    import torch
    from mteb import MTEB
    import json
    from sentence_transformers.models import Normalize

    orig_has_norm = model._last_module().__module__.endswith(".Normalize")
    concat_norm = getattr(model, "concat_norm", None)

    if normed is not None:
        if normed:
            if not orig_has_norm:
                model = model.append(Normalize())
        else:
            if orig_has_norm:
                model.pop(-1)
    else:
        normed = orig_has_norm

    if cache_folder is None:
        cache_folder = "cache_dir"
    if output_folder is None:
        output_folder = "results"

    model_path = f"{cache_folder}/{model_id.replace('/', '_')}"
    steps = 0

    if hasattr(model, "commit_info_json_path") and model.commit_info_json_path is not None:
        commit_info_path = model.commit_info_json_path
    else:
        commit_info_path = f"{model_path}/commit-info.json"

    try:
        commit_info = json.load(open(commit_info_path, "r"))
    except FileNotFoundError:
        commit_info = None

    if commit_info is not None:
        try:
            commit_message = json.loads(commit_info["commit_message"])
            steps = commit_message["step"]
        except Exception as e:
            steps = commit_info["commit_message"].split()[4]

    m_id = model_id.split('/')[1]

    feature = "concat_norm" if concat_norm else ("normed" if normed else "unnormed")

    if use_normed_tasks:
        model.normed_tasks = NORMED_TASKS
        feature = "normed_tasks"

    if expected_step is not None:
        assert int(steps) == int(expected_step)

    results_dir = f"{output_folder}/{m_id}-{steps:06}-feature={feature}"

    if use_normed_tasks:
        os.makedirs(results_dir, exist_ok=True)

        with open(os.path.join(results_dir, "normed_tasks.json"), "w") as fl:
            json.dump(NORMED_TASKS, fl)

    if task_types is None:
        task_types = ["STS", "PairClassification", "Reranking", "Retrieval", "Classification", "Summarization", "BitextMining", "Clustering"]

    scores = {}

    task_set = {
        'STS': (TASK_LIST_STS, compute_sts),
        'PairClassification': (TASK_LIST_PAIR_CLASSIFICATION, compute_pair),
        'Reranking': (TASK_LIST_RERANKING, compute_reranking),
        'Retrieval': (TASK_LIST_RETRIEVAL, compute_retrieval),
        'Classification': (TASK_LIST_CLASSIFICATION, compute_classification),
        'Clustering': (TASK_LIST_CLUSTERING, compute_clustering),
        'Summarization': (TASK_LIST_SUMMARIZATION, compute_summarization),
    }

    for task_type in task_types:
        print(f"Running task type: {task_type}")
        tasks = TASK_LIST_GROUP.get(task_type)
        if tasks is None:
            continue

        for task in tasks:
            if use_normed_tasks:
                model.task = task
            eval_splits = ["dev"] if task == "MSMARCO" else ["test"]
            evaluation = MTEB(task_langs=["en"], tasks=[task])
            print(evaluation.run(model, output_folder=f"{results_dir}/{task_type}", eval_splits=eval_splits, batch_size=batch_size))
            torch.cuda.empty_cache()

        try:
            _, score_func = task_set[task_type]
            scores[task_type] = score_func(tasks, output_folder=f"{results_dir}/{task_type}", agg=True)
            print(f"{task_type}: {scores[task_type]}")
        except Exception as e:
            print(e)
            scores[task_type] = None

        if scores:
            with open(f"{results_dir}/scores.json", "w") as fOut:
                print(scores)
                json.dump(scores, fOut, sort_keys=True, indent=2)

    return results_dir

# model_id = "avsolatorio/MixEmbedLarge-large-20240116235934-best"; model = gist_embed.MixSentenceTransformer(model_id, cache_folder="./cache_dir", device="cuda")
# po = model.encode(["Hello world", "World", "poverty", "cimate change", "co2", "methane", "hola bota de car", "what is the name of a musical instrument?"], output_value=None)


# model_id = "avsolatorio/MixEmbedFinetune-base-20240122132422-best"; model = gist_embed.base.EncoderSentenceTransformer(model_id, cache_folder="./cache_dir", device="cuda")

def main(run_type, model_id, tasks=None, cache_folder="./cache_dir", output_folder=None, device=None, revision=None, batch_size=32, normed=None, use_normed_tasks=False, expected_step=None, rev_task=False):
    import gist_embed

    model = gist_embed.base.EncoderSentenceTransformer(model_id, cache_folder=cache_folder, device=device, revision=revision)

    if run_type == "run_all_mteb":
        run_all_mteb(
            model,
            model_id,
            task_types=None,
            cache_folder=cache_folder,
            output_folder=output_folder,
            eval_splits=["test"],
            compute_score=True,
            batch_size=batch_size,
            normed=normed,
            use_normed_tasks=use_normed_tasks,
            expected_step=expected_step,
        )
    elif run_type == "run_mteb":
        if tasks:
            tasks = globals().get(tasks, TASK_NAMES_EVAL_TIME)
        else:
            tasks = TASK_NAMES_EVAL_TIME
        run_mteb(
            model,
            model_id,
            tasks=tasks,
            cache_folder=cache_folder,
            output_folder=output_folder,
            eval_splits=["test"],
            compute_score=True,
            batch_size=batch_size,
            normed=normed,
            use_normed_tasks=use_normed_tasks,
            expected_step=expected_step,
            rev_task=rev_task,
        )
    else:
        raise ValueError

if __name__ == "__main__":
    """
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/MixEmbedFinetune-base-20240125001825-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=7e7a04257460d3e2797dca021378092e28e620a5 \
    --batch_size=32 \
    --use_normed_tasks


poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/20-100-11-1-0-1-0-0-cls-normed-1024-1024_GIST_bge-base-en-v1.5_1024-20240128194346-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=82590753992c301a33112481e9dd540c03c3380f \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_LIST_PRIORITY



# from evaluation.mteb_eval import *
# import gist_embed
# model_id = "avsolatorio/MixEmbedFinetune-base-20240125001825-latest"; model = gist_embed.base.EncoderSentenceTransformer(model_id, cache_folder="./cache_dir", device="cpu", revision="7e7a04257460d3e2797dca021378092e28e620a5")
# run_mteb(model, model_id, tasks=TASK_NAMES_EVAL_TIME, compute_score=True, batch_size=32, use_normed_tasks=True)
    """


# tasks=None, cache_folder="./cache_dir", device=None, revision=None, batch_size=32, normed=None, use_normed_tasks=True)

    fire.Fire(main)

# poetry run python evaluation/mteb_eval.py --run_type run_mteb \
#     --model_id avsolatorio/MixEmbedFinetune-base-20240125001825-latest \
#     --cache_folder=./cache_dir \
#     --device=cpu \
#     --batch_size=32 \
#     --use_normed_tasks

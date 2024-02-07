## Running evaluation in one machine

```bash
# Terminal 1
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-0-1-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240130214500-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=82642f190fbcb2403e35f78ac628b2a36b8bcce4 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# Terminal 2
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-0-1-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240130214500-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=82642f190fbcb2403e35f78ac628b2a36b8bcce4 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```


```bash
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-100-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240129004245-best \
    --cache_folder=./cache_dir \
    --device=cuda \
    --revision=87ad6fa0a44e5d51f94c6036b064812e5aabdc7b \
    --batch_size=256 \
    --normed=True


poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-100-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240129004245-best \
    --cache_folder=./cache_dir \
    --device=cuda \
    --revision=87ad6fa0a44e5d51f94c6036b064812e5aabdc7b \
    --batch_size=256 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```



```bash
Step 248000
# Terminal 1
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-0-1-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240130214500-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=62d9aa36c677aa6261c95ca761edf5d4a3bcf5c3 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# Terminal 2
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-0-1-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240130214500-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=62d9aa36c677aa6261c95ca761edf5d4a3bcf5c3 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```


```bash
Step 40500
# Terminal 1
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/30-100-11-1-2-2-0-0-mean-normed-384-512_GIST_st_all-MiniLM-L6-v2-20240201190120-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=ec5969403a70300965681808e052e4a5329f7815 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# Terminal 2
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/30-100-11-1-2-2-0-0-mean-normed-384-512_GIST_st_all-MiniLM-L6-v2-20240201190120-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=ec5969403a70300965681808e052e4a5329f7815 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```




```bash
Step 20000
# Terminal 1
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-500-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240201200915-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=378cc473fb1835c4e8677e85b0db5e91a953db32 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# Terminal 2
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-500-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240201200915-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=378cc473fb1835c4e8677e85b0db5e91a953db32 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```



```bash
Step 105500
# Terminal 1
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240131033903-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=356d5db79a86dfdfd4be7fc27cfa03c68a93f6c9 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# Terminal 2
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/00-400-11-1-3-2-0-0-cls-normed-768-512_GIST_BAAI_bge-base-en-v1.5-20240131033903-latest \
    --cache_folder=./cache_dir \
    --device=cuda:0 \
    --revision=356d5db79a86dfdfd4be7fc27cfa03c68a93f6c9 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True
```



avsolatorio/30-100-11-1-2-2-0-0-mean-normed-384-512_GIST_st_all-MiniLM-L6-v2-20240201131522-best










JHUB:

CUDA_VISIBLE_DEVICES=0 bash experiments/next/30-500-11-1-2-2-0-0-mean-normed-384-512_run_finetune_experiment.sh runs/30-500-11-1-2-2-0-0-mean-normed-384-512_GIST_st_all-MiniLM-L6-v2-20240203022317/checkpoint-74000


CUDA_VISIBLE_DEVICES=1 bash experiments/next/30-100-10-1-2-2-0-0-mean-normed-384-512_run_finetune_experiment.sh




avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest


8a5bf0cf47fbc8c0c966c8cfbd99145e4c1dea47



# 103500
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=cuda \
    --revision=8a5bf0cf47fbc8c0c966c8cfbd99145e4c1dea47 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME

# 103500
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=cuda \
    --revision=8a5bf0cf47fbc8c0c966c8cfbd99145e4c1dea47 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True




# 107500
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=a8efb648c45a17ab61426c152ddf6d1295815ebe \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True






# 110500
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=c1eb2e40acc3eb0497f25f5d2627e38c656088c8 \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True







# 95500
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=d8af16c1a91b8f22a27238e25ed8ac0136a94b79 \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True



# 61000
poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=b86f66b5ab0f02b5efd37248b8cc054f35b61133 \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True



poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=cuda \
    --revision=b86f66b5ab0f02b5efd37248b8cc054f35b61133 \
    --batch_size=128 \
    --normed=True \
    --tasks=TASK_NAMES_EVAL_TIME \
    --rev_task=True







poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=8a5bf0cf47fbc8c0c966c8cfbd99145e4c1dea47 \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_LIST_RERANKING




poetry run python evaluation/mteb_eval.py --run_type run_mteb \
    --model_id avsolatorio/50-100-12-1-1-2-0-0-cls-normed-1024-512_GIST_WhereIsAI_UAE-Large-V1-20240204235340-latest \
    --cache_folder=./cache_dir \
    --device=mps \
    --revision=b86f66b5ab0f02b5efd37248b8cc054f35b61133 \
    --batch_size=32 \
    --normed=True \
    --tasks=TASK_LIST_RERANKING
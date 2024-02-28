# 01-600-11-1-2-2-0-0-cls-normed-384-512_run_finetune_experiment.sh
export XX=01  # model id
export YYY=600  # dataset id
export ZZ=11  # loss type id
export a=1  # learning rate id
export b=2  # batch size id
export c=2  # epochs id
export d=0  # warmup ratio id
export e=0  # cl temperature id
export MODEL_NAME_OR_PATH=BAAI/bge-small-en-v1.5
export AUTO_MODEL_POOLING=cls
export LOSS_TYPE=guided
export GUIDE_MODEL_NAME_OR_PATH=WhereIsAI/UAE-Large-V1
export GIST_MODEL_TYPE=GIST_${MODEL_NAME_OR_PATH//\//_}
export MEDI_DATA_NAME=avsolatorio/medi-data-mteb-covid-bing-query-gpt4-avs_triplets
export OUTPUT_DIM=384
export MEDI_DATA_REVISION=7612b607f896cbf5d769dbe838ac83ce0807056b
export NUM_TRAIN_EPOCHS=40
export CL_TEMPERATURE=0.01
export WARMUP_RATIO=0.1
export LEARNING_RATE=5e-6
export PER_DEVICE_BATCH_SIZE=16
export MAX_SOURCE_LENGTH=512
export WANDB_WATCH=all
export GIST_NORMALIZE=true
export WANDB_PROJECT=${GIST_MODEL_TYPE}
export SCRIPT_ID=${XX}-${YYY}-${ZZ}-${a}-${b}-${c}-${d}-${e}-${AUTO_MODEL_POOLING}-normed-${OUTPUT_DIM}-${MAX_SOURCE_LENGTH}
export OUTPUT_DIR=${SCRIPT_ID}_${GIST_MODEL_TYPE}-$(date +"%Y%m%d%H%M%S")
export RUN_NAME=r_${OUTPUT_DIR}

export OUTPUT_DIR=runs/${OUTPUT_DIR}

echo ${RUN_NAME}

if [[ -z "$1" ]]; then
    CHECKPOINT=None
else
    CHECKPOINT=$1;
fi

poetry run python train_finetune.py \
--model_name_or_path ${MODEL_NAME_OR_PATH} \
--preprocessing_num_workers 16 \
--gist_auto_model_pooling ${AUTO_MODEL_POOLING} \
--gist_script_id ${SCRIPT_ID} \
--gist_loss_type ${LOSS_TYPE} \
--gist_guide_model_name_or_path ${GUIDE_MODEL_NAME_OR_PATH} \
--gist_output_dim ${OUTPUT_DIM} \
--gist_normalize ${GIST_NORMALIZE} \
--gist_medi_data_name ${MEDI_DATA_NAME} \
--gist_medi_data_name_revision ${MEDI_DATA_REVISION} \
--gist_cl_temperature ${CL_TEMPERATURE} \
--callback_save_to_hub \
--callback_hub_organization avsolatorio \
--callback_hub_private \
--callback_hub_exist_ok \
--callback_hub_replace_model_card \
--callback_hub_run_as_future \
--output_dir ${OUTPUT_DIR} \
--max_source_length ${MAX_SOURCE_LENGTH} \
--num_train_epochs ${NUM_TRAIN_EPOCHS} \
--logging_steps 500 \
--save_steps 500 \
--warmup_ratio ${WARMUP_RATIO} \
--learning_rate ${LEARNING_RATE} \
--overwrite_output_dir \
--resume_from_checkpoint ${CHECKPOINT} \
--per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
--gradient_accumulation_steps 1 \
--cache_dir medi-data \
--hub_strategy checkpoint \
--push_to_hub true \
--hub_private_repo \
--report_to wandb \
--run_name ${RUN_NAME} \
--save_safetensors false \
--gradient_checkpointing true \
--save_total_limit 1 \
--bf16 \
"$@"

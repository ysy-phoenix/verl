#!/bin/bash

# MAIN CONFIG
export VLLM_USE_V1=1
MAX_EPOCHS=8
TRAIN_DATASET=code-r1
TEST_DATASET=r1
MODEL_NAME=Qwen2.5-Coder-7B-Instruct
MODEL_PATH="$HOME/models/$MODEL_NAME"
ROLLOUT_N_SAMPLE=4
TRAIN_BATCH_SIZE=8
MINI_BATCH_SIZE=8
MICRO_BATCH_PER_GPU=8
LOG_PROB_MICRO_BATCH_SIZE=64
TP_SIZE=1
EXPERIMENT_NAME=$MODEL_NAME-$TRAIN_DATASET-${TRAIN_BATCH_SIZE}-${ROLLOUT_N_SAMPLE}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/$TRAIN_DATASET/train.parquet \
    data.val_files=$HOME/data/$TEST_DATASET/test.parquet \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=3072 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N_SAMPLE \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$LOG_PROB_MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb','console'] \
    trainer.project_name='qwen-code-r1' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.nnodes=1 \
    trainer.default_local_dir=$HOME/checkpoints/${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=4 \
    trainer.save_freq=256 \
    trainer.test_freq=32 \
    trainer.total_epochs=$MAX_EPOCHS \
    reward_model.reward_manager=prime $@ 2>&1 | tee grpo.log


# step 120s
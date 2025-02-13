#!/bin/bash
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=240:00:00
#SBATCH --job-name=sft
#SBATCH --output slurm/%j.out

ulimit -n 64000
source ~/.bashrc
conda activate zero2
wandb online

echo $DATA_DIR
echo $EXP_NAME
echo $SAVE_DIR
pwd
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
       	-m verl.trainer.fsdp_sft_trainer \
	data.train_files=$DATA_DIR/train.parquet \
	data.val_files=$DATA_DIR/test.parquet \
    	data.prompt_key=extra_info \
	data.response_key=extra_info \
	data.max_length=5012 \
        +data.prompt_dict_keys=['question'] \
        +data.response_dict_keys=['answer'] \
	data.micro_batch_size=4 \
	model.partial_pretrain=Qwen/Qwen2.5-0.5B-Instruct \
	trainer.project_name=qwen0.5b-sft \
	trainer.experiment_name=$EXP_NAME \
	trainer.total_epochs=null \
	trainer.total_steps=100 \
	trainer.logger=['console','wandb'] \
	trainer.default_local_dir=$SAVE_DIR \

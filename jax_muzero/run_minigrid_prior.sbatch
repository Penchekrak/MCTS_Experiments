#!/bin/bash
#SBATCH --job-name=run_minigrid_prior            # Название задачи
#SBATCH --error=run_minigrid_prior.err        # Файл для вывода ошибок
#SBATCH --output=run_minigrid_prior.log       # Файл для вывода результатов
#SBATCH --time=48:00:00            # Максимальное время выполнения
#SBATCH --nodes=1                  # Требуемое кол-во узлов
#SBATCH --gpus=1                   # Требуемое кол-во GPU
#SBATCH --cpus-per-task=8          # Требуемое кол-во CPU

export CUDA_DEVICE_ORDER='PCI_BUS_ID'
export CUDA_VISIBLE_DEVICES=0

#python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force --num_gpus 2 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 --p_mcts_num 1 --seed 42 --use_priority --use_max_priority --amp_type 'torch_amp' --info 'EfficientZero-V1' --save_video
#python3 main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --force --num_gpus 2 --num_cpus 32 --cpu_actor 8 --gpu_actor 8 --p_mcts_num 1 --seed 42 --use_priority --use_max_priority --amp_type 'torch_amp' --info 'EfficientZero-V1' --save_video
module load Python/Anaconda_v11.2021
source activate mcts
python minigrid_prior.py
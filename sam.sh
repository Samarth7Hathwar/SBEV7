#!/bin/bash
#SBATCH --job-name=SBEV7
#SBATCH --partition=A100-PCI,A100-40GB,A100-80GB
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=150G
#SBATCH --time=72:00:00
#SBATCH --output=logs/sbev7_train_%j.log
#SBATCH --error=logs/sbev7_train_%j.err

srun \
  --ntasks=4 \
  --gpus-per-task=1 \
  --cpus-per-task=10 \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,/ds-av:/ds-av,/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck:/workspace,/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/outputs:/outputs,`pwd`:`pwd` \
  --container-image=/netscratch/kpatil/container/SparseBEV.sqsh \
  --container-workdir=/workspace \
  bash -c "
  pip install openmim &&
  mim install mmcv-full==1.6.0 &&
  pip install -r requirements.txt || true &&
  torchrun --nproc_per_node=4 train.py configs/r50_nuimg_704x256.py"
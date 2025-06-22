#!/bin/bash
#SBATCH --job-name=sbev_train
#SBATCH --partition=A100-PCI,batch,A100-40GB,A100-80GB,A100-RP,RTXA6000,RTX3090
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=Results/SBEV_Train_E0.out
#SBATCH --error=Results/SBEV_Train_E0.err

srun \
  --container-image=/netscratch/kpatil/container/SparseBEV.sqsh \
  --container-name=sparsebev_container \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,/ds-av:/ds-av,\
/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck:/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck,\
`pwd`:`pwd` \
  --container-workdir=`pwd` \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  bash sparsebev_train.sh

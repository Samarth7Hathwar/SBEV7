#!/bin/bash
#SBATCH --job-name=SBEV7_Train_E0
#SBATCH --partition=A100-PCI,batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem-per-gpu=40G
#SBATCH --time=3-00:00:00
#SBATCH --output=/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/SBEV_Train_E0.out
#SBATCH --error=/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/SBEV_Train_E0.err

srun -K \
  --container-mounts=/netscratch:/netscratch,/ds:/ds,/ds-av:/ds-av,/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck:/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck,/netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/outputs:/outputs,`pwd`:`pwd` \
  --container-image=/netscratch/kpatil/container/SparseBEV.sqsh \
  --container-workdir=`pwd` \
  --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
  bash /netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/sparsebev_train.sh
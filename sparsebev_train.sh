#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
echo "*********************************Move to SABEV directory*********************************"
cd /netscratch/hathwar/guided_research/SparseBEV7_Radar_S5_withNeck/

echo "*********************************build*********************************"
#python setup.py develop
# conda env create --file CRN.yaml
# source /opt/conda/etc/profile.d/conda.sh      #sabev
# source ~/anaconda3/etc/profile.d/conda.sh     #crn
source ~/miniconda/etc/profile.d/conda.sh     #sparsebev trial
conda activate sparsebev
pip install fvcore
pip install einops
pip install iopath


echo "*********************************training model*********************************"
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi -L  # List all GPUs
echo $CUDA_VISIBLE_DEVICES  # Verify GPU visibility
export TORCH_DISTRIBUTED_DEBUG=DETAIL

export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 train.py --config configs/r50_nuimg_704x256.py 
###--override resume_from=outputs/SparseBEV/2025-04-16/17-31-42/ep5.pth 
echo "*********************************testing model*********************************"
# SBEV7 --> bs-1 = 7.7GB, bs-2 = 11.7GB, RTXA6000 bs-6 per gpu
# make sure to scale worker per gpu appropriatly
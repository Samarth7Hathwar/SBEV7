#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
echo "*********************************Move to SABEV directory*********************************"
cd /netscratch/kpatil/SparseBEV/SparseBEV7_Radar_S5_withNeck/

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

export CUDA_VISIBLE_DEVICES=0
python val.py --config configs/r50_nuimg_704x256.py --weights outputs/SparseBEV/2025-05-19/18-28-43/ep21.pth

echo "*********************************testing model*********************************"
                                                        			#MAP    #NDS
        # GITHUB (CAMERA ONLY) E24                      			43.2     54.5
        # (CAMERA + RADAR) BELOW 

###SBEV7_140x88_64x176_interpolate_E0
###SparseBEV7_Radar_S5_withNeck/outputs/SparseBEV/2025-05-19/18-28-43/ep21.pth  40.3    51.4

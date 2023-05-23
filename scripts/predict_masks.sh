#!/bin/bash
#SBATCH -J pred_sam
#SBATCH -o pred_sam.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

INPUT="/mnt/external.data/TowbinLab/igheor/20230403_NikonEclipseTi2_Exp29/raw/20230403_IG7_wBT186_IAA_20C/WellF06_Channel470_nm,575,DIA_Seq0030.tiff"
OUTPUT="/mnt/external.data/TowbinLab/igheor/20230403_NikonEclipseTi2_Exp29/analysis/test_sam_masks/"

source ~/env_directory/segment_anything/bin/activate
python3 amg.py --input "$INPUT" --output "$OUTPUT" --model-type "vit_h" --checkpoint "../models/sam_vit_h_4b8939.pth" --points-per-side 32 --points-per-batch 32 --pred-iou-thresh 0.86 --stability-score-thresh 0.90 --channel 1
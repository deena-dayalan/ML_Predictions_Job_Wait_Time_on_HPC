#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=FeatureEngineering
#SBATCH --time=23:59:59
#SBATCH -N 1
#SBATCH -n 128
#SBATCH --exclusive
#SBATCH --output=%j.output
#SBATCH --error=%j.error

## load the anaconda package:
module load anaconda3/2021.05
## activate the environment
source activate conda3.8

## run the python script
python /home/d.dasarathan/project_sml/featureEngineering.py
echo "task completed" > fea.log

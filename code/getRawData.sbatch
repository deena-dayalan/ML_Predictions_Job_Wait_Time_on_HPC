#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=getRawData
#SBATCH --time=24:00:00
#SBATCH -N 1
#SBATCH -n 100
#SBATCH --mem=40G
#SBATCH --output=%j.output
#SBATCH --error=%j.error

module load anaconda3/2021.05
#sacct  --starttime=2021-08-12 --endtime=2021-08-13 -p --delimiter ',' -all > /home/d.dasarathan/project_sml/rawSacctData.csv
sacct  --starttime=2021-08-12 --endtime=2021-08-13 -p -all > /home/d.dasarathan/project_sml/rawSacctData

python toPickle.py /home/d.dasarathan/project_sml/rawSacctData.csv

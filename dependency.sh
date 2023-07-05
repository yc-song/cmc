#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P1

iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable train_biencoder_sweep.sh)
jobid=321937
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --dependency=$dependency run.sh)
    dependency=",${dependency}afterany:${jobid}"
done
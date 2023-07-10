#!/bin/bash

#SBATCH --job-name=dependency
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --time=0-12:00:00
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P1

iterations=2 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable reranker_restart.sh)
jobid=323420
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency reranker_restart.sh)
done
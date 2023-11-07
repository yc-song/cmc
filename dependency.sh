#!/bin/bash
#SBATCH --job-name=vt3udjrx
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1
#SBATCH --partition=P2
#SBATCH --comment=pytorch


iterations=5 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable run.sh)

# jobid=$(sbatch --parsable reranker_extend.sh)
jobid=105928
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency reranker_extend.sh)
done
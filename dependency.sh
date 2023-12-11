#!/bin/bash
#SBATCH --job-name=vt3udjrx
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1


iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable run.sh)

<<<<<<< HEAD
jobid=$(sbatch --parsable reranker_extend.sh)
# jobid=252146
=======
# jobid=$(sbatch --parsable reranker_extend.sh)
jobid=257200
>>>>>>> 07c004da80bb8a89038b95157afad352f8580f47
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency reranker_extend.sh)
done

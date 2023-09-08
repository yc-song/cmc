#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=extend_freeze
=======

#SBATCH --job-name=dependency
>>>>>>> d14dc70d79c589d1de2e80725b7182d1d82f1ec7
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
#SBATCH --mem=8000MB
#SBATCH --cpus-per-task=1
<<<<<<< HEAD
#SBATCH --partition=cas_v100nv_8
#SBATCH --comment pytorch


iterations=1 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable run.sh)

# jobid=$(sbatch --parsable reranker_restart.sh $1 $2)
jobid=233115
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency reranker_extend.sh $1 $2)
=======
#SBATCH --partition=P1
#SBATCH -o ./slurm_output/%x_%j.out
#SBATCH -e ./slurm_output/%x_%j.err
iterations=5 # 총 몇 번이나 연속으로 돌릴 것인지
# jobid=$(sbatch --parsable run.sh)

jobid=$(sbatch --parsable reranker_fixed_weight.sh)
# jobid=5850
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
    jobid=$(sbatch --parsable --dependency=$dependency reranker_fixed_weight.sh)
>>>>>>> d14dc70d79c589d1de2e80725b7182d1d82f1ec7
done

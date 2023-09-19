#!/bin/bash
<<<<<<< HEAD
#SBATCH --job-name=extend_freeze
=======

<<<<<<< HEAD
#SBATCH --job-name=dependency
>>>>>>> d14dc70d79c589d1de2e80725b7182d1d82f1ec7
=======
#SBATCH --job-name=4yiwnqcn
>>>>>>> 68000b0f97e8e15db906a5a1b6b51885ba23641f
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

jobid=$(sbatch --parsable reranker_diff_lr.sh)
# jobid=13941
for((i=0; i<$iterations; i++)); do           
    dependency="afterany:${jobid}"
    echo "dependency: $dependency"
<<<<<<< HEAD
    jobid=$(sbatch --parsable --dependency=$dependency reranker_fixed_weight.sh)
>>>>>>> d14dc70d79c589d1de2e80725b7182d1d82f1ec7
=======
    jobid=$(sbatch --parsable --dependency=$dependency reranker_diff_lr.sh)
>>>>>>> 68000b0f97e8e15db906a5a1b6b51885ba23641f
done

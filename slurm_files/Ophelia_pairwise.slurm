#!/bin/bash
#SBATCH --job-name=Ising
#SBATCH --mail-type=ALL
#SBATCH --partition day
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10gb
#SBATCH --time=10:00:00

cd ${SLURM_SUBMIT_DIR}
echo "I'm the best"

module load miniconda
source activate base
conda activate gpu2026
cd ../src/

python Ophelia_pairwise.py

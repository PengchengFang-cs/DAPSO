#!/bin/bash
#SBATCH -p swarm_h100
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=96:00:00
#SBATCH --job-name=train_vmamba
#SBATCH --output=logs/fastmri_8x_p2_6x2_mambarecon_fourier_1121.log
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=80G

source ~/.bashrc
conda activate Mamba_IR

cd /scratch/pf2m24/projects/MRIRecon/Dual_Axis/code

echo "CPUS ON NODE: $SLURM_CPUS_ON_NODE"
echo "JOB CPUS PER NODE: $SLURM_JOB_CPUS_PER_NODE"
echo "CPUS PER TASK: $SLURM_CPUS_PER_TASK"

export PYTHONPATH=/scratch/pf2m24/projects/MRIRecon/MambaReconV3/code:$PYTHONPATH

torchrun --nproc_per_node=2 --master_port=29513 train_scan_out_8_fastmri_best_ddp_resume_fix.py --batch_size 8 --name fastmri_8x_p2_6x2_mambarecon_fourier_1121
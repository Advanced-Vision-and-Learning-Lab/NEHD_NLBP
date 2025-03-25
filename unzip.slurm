#!/bin/bash
#
##NESSARY JOB SPECIFICATIONS`  
#SBATCH --job-name=unzip              # Set the job name to Job
#SBATCH --time=08:00:00              # Set the wall clock limit to HH:MM:SS
#SBATCH --nodes=1                    # How many nodes to request, this should almost always be 1
#SBATCH --cpus-per-task=1            # How many cpus to request per task, when using with lightning this should be > --num_data_workers hyperparameters
#SBATCH --mem=32G                    # Request RAM that is per node 8 to 16GB is usually sufficient, this depends on the size of the dataset
#SBATCH --output=EHD_Exp_Dilation2       # Set the name of the output file, use a unique name for each job
#SBATCH --partition=gpu              # Specify partition to submit job to, dont change this
#SBATCH --gres=gpu:0                 # Specify GPU(s) per node, 2 A100 gpu; Use more than 1 if you are running out of memory due to the GPU


# Set up Python
module purge
module load GCCcore/13.2.0 Python/3.11.5
ml CUDA

# Go to the directory and activate the venv
/scratch/user/k1411130/NEHD_NLBP
source venv/bin/activate

cd Datasets/
unzip PRMI_official.zip

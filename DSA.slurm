#!/bin/bash
#
##NECESSARY JOB SPECIFICATIONS`  
#SBATCH --job-name=test_dsa              # Set the job name to Job
#SBATCH --time=03:30:00              # Set the wall clock limit to HH:MM:SS
#SBATCH --nodes=1                    # How many nodes to request, this should almost always be 1
#SBATCH --cpus-per-task=1            # How many cpus to request per task, when using with lightning this should be > --num_data_workers hyperparameters
#SBATCH --mem=16G                    # Request RAM that is per node 8 to 16GB is usually sufficient, this depends on the size of the dataset
#SBATCH --output=DSA_23       # Set the name of the output file, use a unique name for each job
#SBATCH --partition=gpu              # Specify partition to submit job to, dont change this
#SBATCH --gres=gpu:1                 # Specify GPU(s) per node, 2 A100 gpu; Use more than 1 if you are running out of memory due to the GPU


# Set up Python
module purge
module load GCCcore/13.2.0 Python/3.11.5
ml CUDA

# Go to the directory and activate the venv
/scratch/user/k1411130/NEHD_NLBP
source venv/bin/activate

#python demo.py --num_epochs=100 --feature=DSA --no-histogram --data_selection=1
python demo.py --num_epochs=100 --feature=DSA --no-histogram --data_selection=2
python demo.py --num_epochs=100 --feature=DSA --no-histogram --data_selection=3


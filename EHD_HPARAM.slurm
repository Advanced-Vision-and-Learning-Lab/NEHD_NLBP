#!/bin/bash
#
##NESSARY JOB SPECIFICATIONS`  
#SBATCH --job-name=EHD_Exp              # Set the job name to Job
#SBATCH --time=18:00:00              # Set the wall clock limit to HH:MM:SS
#SBATCH --nodes=1                    # How many nodes to request, this should almost always be 1
#SBATCH --cpus-per-task=1            # How many cpus to request per task, when using with lightning this should be > --num_data_workers hyperparameters
#SBATCH --mem=32G                    # Request RAM that is per node 8 to 16GB is usually sufficient, this depends on the size of the dataset
#SBATCH --output=EHD_Exp_Dilation2       # Set the name of the output file, use a unique name for each job
#SBATCH --partition=gpu              # Specify partition to submit job to, dont change this
#SBATCH --gres=gpu:1                 # Specify GPU(s) per node, 2 A100 gpu; Use more than 1 if you are running out of memory due to the GPU


# Set up Python
module purge
module load GCCcore/13.2.0 Python/3.11.5
ml CUDA

# Go to the directory and activate the venv
/scratch/user/k1411130/NEHD_NLBP
source venv/bin/activate


# Structural- kernel size (3x3, 5x5, 7x7); Statistical- aggregation kernel (3x3, 5x5, 7x7)

# 3,3
python demo.py --num_epochs=100 --folder=Saved_Models/13 --dilation=1 --window_size 3 3
python View_Results.py --num_epochs=100 --folder=Saved_Models/13 --dilation=1 --window_size 3 3

# 3,5
python demo.py --num_epochs=100 --folder=Saved_Models/15 --dilation=1 --window_size 5 5 
python View_Results.py --num_epochs=100 --folder=Saved_Models/15 --dilation=1 --window_size 5 5

# 3,7
python demo.py --num_epochs=100 --folder=Saved_Models/17 --dilation=1 --window_size 7 7 
python View_Results.py --num_epochs=100 --folder=Saved_Models/17 --dilation=1 --window_size 7 7

# 5,3
python demo.py --num_epochs=100 --folder=Saved_Models/23 --dilation=2 --window_size 3 3 
python View_Results.py --num_epochs=100 --folder=Saved_Models/23 --dilation=2 --window_size 3 3

# 5,5
python demo.py --num_epochs=100 --folder=Saved_Models/25 --dilation=2 --window_size 5 5
python View_Results.py --num_epochs=100 --folder=Saved_Models/25 --dilation=2 --window_size 5 5

# 5,7
python demo.py --num_epochs=100 --folder=Saved_Models/27 --dilation=2 --window_size 7 7
python View_Results.py --num_epochs=100 --folder=Saved_Models/27 --dilation=2 --window_size 7 7

# 7,3
python demo.py --num_epochs=100 --folder=Saved_Models/33 --dilation=3 --window_size 3 3 
python View_Results.py --num_epochs=100 --folder=Saved_Models/33 --dilation=3 --window_size 3 3

# 7,5
python demo.py --num_epochs=100 --folder=Saved_Models/35 --dilation=3 --window_size 5 5
python View_Results.py --num_epochs=100 --folder=Saved_Models/35 --dilation=3 --window_size 5 5

# 7,7
python demo.py --num_epochs=100 --folder=Saved_Models/37 --dilation=3 --window_size 7 7
python View_Results.py --num_epochs=100 --folder=Saved_Models/37 --dilation=3 --window_size 7 7

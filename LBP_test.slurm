#!/bin/bash
#
##NECESSARY JOB SPECIFICATIONS`  
#SBATCH --job-name=LBP_Exp              # Set the job name to Job
#SBATCH --time=16:00:00              # Set the wall clock limit to HH:MM:SS
#SBATCH --nodes=1                    # How many nodes to request, this should almost always be 1
#SBATCH --cpus-per-task=4            # How many cpus to request per task, when using with lightning this should be > --num_data_workers hyperparameters
#SBATCH --mem=128G                    # Request RAM that is per node 8 to 16GB is usually sufficient, this depends on the size of the dataset
#SBATCH --output=LBP_Exp_test       # Set the name of the output file, use a unique name for each job
#SBATCH --partition=gpu              # Specify partition to submit job to, dont change this
#SBATCH --gres=gpu:1                 # Specify GPU(s) per node, 2 A100 gpu; Use more than 1 if you are running out of memory due to the GPU


# Set up Python
module purge
module load GCCcore/13.2.0 Python/3.11.5
ml CUDA

# Go to the directory and activate the venv
/scratch/user/k1411130/NEHD_NLBP
source venv/bin/activate


# NLBP: Structural- dilation size (1, 2, 3); Statistical- aggregation kernel  (3x3, 5x5, 7x7)

# NLBP: Structural- dilation size (1, 2, 3); Statistical- aggregation kernel (3x3, 5x5, 7x7)

# 1,3
python demo.py --num_epochs=100 --folder=Saved_Models/13 --dilation=1 --window_size 3 3 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/13 --dilation=1 --window_size 3 3 --feature=LBP 

# 1,5
python demo.py --num_epochs=100 --folder=Saved_Models/15 --dilation=1 --window_size 5 5 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/15 --dilation=1 --window_size 5 5 --feature=LBP 

# 1,7
python demo.py --num_epochs=100 --folder=Saved_Models/17 --dilation=1 --window_size 7 7 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/17 --dilation=1 --window_size 7 7 --feature=LBP 

# 2,3
python demo.py --num_epochs=100 --folder=Saved_Models/23 --dilation=2 --window_size 3 3 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/23 --dilation=2 --window_size 3 3 --feature=LBP 

# 2,5
python demo.py --num_epochs=100 --folder=Saved_Models/25 --dilation=2 --window_size 5 5 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/25 --dilation=2 --window_size 5 5 --feature=LBP 

# 2,7
python demo.py --num_epochs=100 --folder=Saved_Models/27 --dilation=2 --window_size 7 7 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/27 --dilation=2 --window_size 7 7 --feature=LBP 

# 3,3
python demo.py --num_epochs=100 --folder=Saved_Models/33 --dilation=3 --window_size 3 3 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/33 --dilation=3 --window_size 3 3 --feature=LBP 

# 3,5
python demo.py --num_epochs=100 --folder=Saved_Models/35 --dilation=3 --window_size 5 5 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/35 --dilation=3 --window_size 5 5 --feature=LBP 

# 3,7
python demo.py --num_epochs=100 --folder=Saved_Models/37 --dilation=3 --window_size 7 7 --feature=LBP 
python View_Results.py --num_epochs=100 --folder=Saved_Models/37 --dilation=3 --window_size 7 7 --feature=LBP 

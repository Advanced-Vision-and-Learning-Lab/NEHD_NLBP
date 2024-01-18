import os
import pandas as pd
import re
import argparse

# Essential Functions
def generate_EHD(path, type, fusion_method):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    print(folders)
    # Init an empty dataframe
    cols = ['Learn_All', 'Fix_All', 'Learn_Kernels', 'Learn_Hist']
    rows = ['EHD_init_Conv_No_Edge', 'EHD_init_Thres_No_Edge', 'Rand_init_Conv_No_Edge', 'Rand_init_Thres_No_Edge']
    df = pd.DataFrame(index=rows, columns=cols)
    # Now we can populate the dataframe
    for folder in folders:
        # Going through the folders in the main folder (ie Fix_All)
        folder_path = os.path.join(path, folder+"/NEHD_Scale_[3, 3]_Dilate_1_"+fusion_method+"_Local")
        column_name = folder
        if column_name not in df.columns:
            continue
        for file in os.listdir(folder_path):
            # Going through the files in the main folders (ie folder in Fix_All)
            folder_path = os.path.join(path, folder+"/NEHD_Scale_[3, 3]_Dilate_1_"+fusion_method+"_Local", file)
            row = file
            try:
                for info in os.listdir(folder_path):
                    # Save the content of the file
                    if info == "Overall_"+type+"_Accuracy.txt":
                        with open(os.path.join(folder_path, info), 'r') as f:
                            content = f.read()
                        match = re.search(r'(\d+\.\d+).*(\d+\.\d+)', content)
                        result = match.group(1) + ',' + match.group(2)
                        result = result.split(',')
                        result = [float(i) for i in result]
                        result = str(round(result[0],2))+"±"+str(round(result[1],2))
                        df.loc[row, column_name] = result
            except:
                print("Error in folder: ", folder_path)

    # Rename the index
    df = df.rename(index={'EHD_init_Conv_No_Edge': 'EHD_Conv',
                        'EHD_init_Thres_No_Edge': 'EHD_Thresh', 
                        'Rand_init_Conv_No_Edge': 'Rand_Conv',
                        'Rand_init_Thres_No_Edge': 'Rand_Thresh'})
    return df

def generate_LBP(path, type, fusion_method):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    print(folders)
    # Init an empty dataframe
    cols = ['Learn_All', 'Fix_All', 'Learn_Kernels', 'Learn_Hist']
    rows = ['LBP_init_Fixed_Base', 'LBP_init_Learn_Base',
            'Rand_init_Fixed_Base', 'Rand_init_Learn_Base']
    df = pd.DataFrame(index=rows, columns=cols)
    # Now we can populate the dataframe
    for folder in folders:
        # Going through the folders in the main folder (ie Fix_All)
        folder_path = os.path.join(path, folder+"/NLBP_Scale_[3, 3]_Dilate_1_"+fusion_method+"_Local")
        column_name = folder
        if column_name not in df.columns:
            continue
        for file in os.listdir(folder_path):
            # Going through the files in the main folders (ie folder in Fix_All)
            folder_path = os.path.join(path, folder+"/NLBP_Scale_[3, 3]_Dilate_1_"+fusion_method+"_Local", file)
            row = file
            for info in os.listdir(folder_path):
                # Save the content of the file
                if info == "Overall_"+type+"_Accuracy.txt":
                    with open(os.path.join(folder_path, info), 'r') as f:
                        content = f.read()
                    match = re.search(r'(\d+\.\d+).*(\d+\.\d+)', content)
                    result = match.group(1) + ',' + match.group(2)
                    result = result.split(',')
                    result = [float(i) for i in result]
                    result = str(round(result[0],2))+"±"+str(round(result[1],2))
                    df.loc[row, column_name] = result

    # Rename the index
    df = df.rename(index={'LBP_init_Fixed_Base': 'LBP_Fixed_Base', 
                            'LBP_init_Learn_Base': 'LBP_Learn_Base',
                            'Rand_init_Fixed_Base': 'Rand_Fixed_Base',
                            'Rand_init_Learn_Base': 'Rand_Learn_Base'})
    return df

# This is the primary control flow
def main(args):
    # Step 1 is to parse the arguments
    feature = args.feature
    dataset = args.dataset
    type = args.type
    fusion_method = args.fusion_method
    
    # Step 2 is to build the path
    path = 'Saved_Models/Paper/Classification/{feature}/{dataset}/'

    # Step 3 is to generate the dataframe based on the feature
    if feature == 'EHD':
        df = generate_EHD(path.format(feature=feature, dataset=dataset), type, fusion_method)
        print("Overall Results for EHD ", type)
        print(df)
    elif feature == 'LBP':
        df = generate_LBP(path.format(feature=feature, dataset=dataset), type, fusion_method)
        print("Overall Results for LBP ", type)
        print(df)

def parse_args():
    parser = argparse.ArgumentParser(description="Print Tabulated Results")
    parser.add_argument('--feature', type=str, default='EHD', help='EHD or LBP')
    parser.add_argument('--dataset', type=str, default='Fashion_MNIST', help='Write it exactly as it is in the folder')
    parser.add_argument('--fusion_method', type=str, default='None', help='None,grayscale, conv' )
    parser.add_argument('--type', type=str, default='Test', help='Type of the result: Test, SVM, XGB, KNN')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)

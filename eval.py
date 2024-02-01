from __future__ import print_function

import numpy as np

import argparse
import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
import matplotlib.pyplot as plt
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import h5py
from utils.eval_utils import *
import matplotlib.pyplot as plt
from itertools import cycle
from random import randint

# Training settings
parser = argparse.ArgumentParser(description='CLAM Evaluation Script')
parser.add_argument('--data_root_dir', type=str, default=None,
                    help='data directory')
parser.add_argument('--results_dir', type=str, default='./results',
                    help='relative path to results folder, i.e. '+
                    'the directory containing models_exp_code relative to project root (default: ./results)')
parser.add_argument('--save_exp_code', type=str, default=None,
                    help='experiment code to save eval results')
parser.add_argument('--models_exp_code', type=str, default=None,
                    help='experiment code to load trained models (directory under results_dir containing model checkpoints')
parser.add_argument('--splits_dir', type=str, default=None,
                    help='splits directory, if using custom splits other than what matches the task (default: None)')
parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', 
                    help='size of model (default: small)')
parser.add_argument('--model_type', type=str, choices=['clam_sb', 'clam_mb', 'mil'], default='clam_sb', 
                    help='type of model (default: clam_sb)')
parser.add_argument('--drop_out', action='store_true', default=False, 
                    help='whether model uses dropout')
parser.add_argument('--k', type=int, default=10, help='number of folds (default: 10)')
parser.add_argument('--k_start', type=int, default=-1, help='start fold (default: -1, last fold)')
parser.add_argument('--k_end', type=int, default=-1, help='end fold (default: -1, first fold)')
parser.add_argument('--fold', type=int, default=-1, help='single fold to evaluate')
parser.add_argument('--micro_average', action='store_true', default=False, 
                    help='use micro_average instead of macro_avearge for multiclass AUC')
parser.add_argument('--split', type=str, choices=['train', 'val', 'test', 'all'], default='test')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal',  'task_2_tumor_subtyping', 'bttf_full_ischemic-time_2cat_256', 'bttf_full_ischemic-time_3cat_256',
                                                 'All-HER2-HE_3cat_256', 'All-HER2-HE_PosvnotPos_256', 'All-HER2-HE_LowvnotLow_256', 'All-HER2-HE_0vnot0_256'])
parser.add_argument('--seed', type=int, default=1, 
                    help='random seed for reproducible experiment (default: 1)')
args = parser.parse_args()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.save_dir = os.path.join('./eval_results', 'EVAL_' + str(args.save_exp_code))
args.models_dir = os.path.join(args.results_dir, str(args.models_exp_code))

os.makedirs(args.save_dir, exist_ok=True)

if args.splits_dir is None:
    args.splits_dir = args.models_dir

assert os.path.isdir(args.models_dir)
assert os.path.isdir(args.splits_dir)

settings = {'task': args.task,
            'split': args.split,
            'save_dir': args.save_dir, 
            'models_dir': args.models_dir,
            'model_type': args.model_type,
            'drop_out': args.drop_out,
            'model_size': args.model_size}

with open(args.save_dir + '/eval_experiment_{}.txt'.format(args.save_exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print(settings)
if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_vs_normal_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=False,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            data_dir= os.path.join(args.data_root_dir, 'tumor_subtyping_resnet_features'),
                            shuffle = False, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= False,
                            ignore=[])



elif args.task == 'bttf_full_ischemic-time_2cat_256':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/breast-time-to-fixation_full_isch_time_rescanOnly.csv',
                            data_dir= os.path.join(args.data_root_dir, 'Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'2hr':0, '40min':0, '24hr':1, '48hr':1},
                            patient_strat= False,
                            ignore=[])
                            
elif args.task == 'bttf_full_ischemic-time_3cat_256':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/breast-time-to-fixation_full_isch_time_rescanOnly.csv',
                            data_dir= os.path.join(args.data_root_dir, 'Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'40min':0, '2hr':1, '24hr':2, '48hr':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'All-HER2-HE_3cat_256':
    args.n_classes=3
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CLAM_Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':2, '3+':2, '2+AMP':2},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'All-HER2-HE_PosvnotPos_256':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CLAM_Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':0, 'Positive':1, '3+':1, '2+AMP':1},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'All-HER2-HE_LowvnotLow_256':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CLAM_Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':0, '3+':0, '2+AMP':0},
                            patient_strat= False,
                            ignore=[])
elif args.task == 'All-HER2-HE_0vnot0_256':
    args.n_classes=2
    dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            data_dir= os.path.join(args.data_root_dir, 'CLAM_Features-256'),
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':1, '3+':1, '2+AMP':1},
                            patient_strat= False,
                            ignore=[])
# elif args.task == 'tcga_kidney_cv':
#     args.n_classes=3
#     dataset = Generic_MIL_Dataset(csv_path = 'dataset_csv/tcga_kidney_clean.csv',
#                             data_dir= os.path.join(args.data_root_dir, 'tcga_kidney_20x_features'),
#                             shuffle = False, 
#                             print_info = True,
#                             label_dict = {'TCGA-KICH':0, 'TCGA-KIRC':1, 'TCGA-KIRP':2},
#                             patient_strat= False,
#                             ignore=['TCGA-SARC'])

else:
    raise NotImplementedError

if args.k_start == -1:
    start = 0
else:
    start = args.k_start
if args.k_end == -1:
    end = args.k
else:
    end = args.k_end

if args.fold == -1:
    folds = range(start, end)
else:
    folds = range(args.fold, args.fold+1)
ckpt_paths = [os.path.join(args.models_dir, 's_{}_checkpoint.pt'.format(fold)) for fold in folds]
folds = [fold for fold in folds if os.path.exists(ckpt_paths[fold])]
print(folds)
ckpt_paths = [path for path in ckpt_paths if os.path.exists(path)]
print(ckpt_paths)
datasets_id = {'train': 0, 'val': 1, 'test': 2, 'all': -1}

if __name__ == "__main__":
    all_results = []
    all_auc = []
    all_acc = []
    rocs = []
    for ckpt_idx in range(len(ckpt_paths)):
        print('working on checkpoint {}...'.format(ckpt_idx))
        if datasets_id[args.split] < 0:
            split_dataset = dataset
        else:
            csv_path = '{}/splits_{}.csv'.format(args.splits_dir, folds[ckpt_idx])
            datasets = dataset.return_splits(from_id=False, csv_path=csv_path)
            split_dataset = datasets[datasets_id[args.split]]
        # model, patient_results, test_error, auc, df  = eval(split_dataset, args, ckpt_paths[ckpt_idx])
        model, patient_results, test_error, auc, df, roc_df  = eval_allPerformance(split_dataset, args, ckpt_paths[ckpt_idx])
        all_results.append(all_results)
        all_auc.append(auc)
        all_acc.append(1-test_error)
        if args.n_classes > 2:
            rocs.append(roc_df.loc[~pd.isna(roc_df['FPR_macro']), ['FPR_macro', 'TPR_macro']])
        else:
            rocs.append(roc_df.loc[~pd.isna(roc_df['FPR']), ['FPR', 'TPR']])
        df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        roc_df.to_csv(os.path.join(args.save_dir, 'fold_{}_ROC.csv'.format(folds[ckpt_idx])), index=False)
    
    
    lw = 2
    color = []
    n = len(ckpt_paths)
    for i in range(n):
        color.append('#%06X' % randint(0, 0xFFFFFF))
    # Plot all ROC curves
    plt.figure()
    colors = cycle(color)
    for i, colr in zip(range(n), colors):
        roc_data = rocs[i]
        if args.n_classes > 2:
            plt.plot(
                roc_data['FPR_macro'],
                roc_data['TPR_macro'],
                color=colr,
                lw=lw,
                label="Fold {0} (area = {1:0.2f})".format(i, all_auc[i]),
            )
        else:
            plt.plot(
                roc_data['FPR'],
                roc_data['TPR'],
                color=colr,
                lw=lw,
                label="Fold {0} (area = {1:0.2f})".format(i, all_auc[i]),
            )
    
    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves across folds")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.save_dir,'ROC across folds.png'), dpi=500)
    # plt.show()
    
    
    final_df = pd.DataFrame({'folds': folds, 'test_auc': all_auc, 'test_acc': all_acc})
    if len(ckpt_paths) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

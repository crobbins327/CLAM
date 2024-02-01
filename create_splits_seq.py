import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping', 
                'bttf_full_ischemic-time', 'bttf_full_ischemic-time_2cat', 'bttf_full_ischemic-time_3cat', 
                'HEROHE_enough_3cat', 'HEROHE_enough_2cat', 'HEROHE_enough_0vnot0',
                'All-HER2-HE_3cat','All-HER2-HE_PosvnotPos','All-HER2-HE_LowvnotLow','All-HER2-HE_0vnot0',
                'Yale-HSHER2_4cat', 'Yale-HSHER2_3cat'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'bttf_full_ischemic-time':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/breast-time-to-fixation_full_isch_time_rescanOnly.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'40min':0, '2hr':1, '24hr':2, '48hr':3},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'bttf_full_ischemic-time_3cat':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/breast-time-to-fixation_full_isch_time_rescanOnly.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'40min':0, '2hr':1, '24hr':2, '48hr':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'bttf_full_ischemic-time_2cat':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/breast-time-to-fixation_full_isch_time_rescanOnly.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'2hr':0, '40min':0, '24hr':1, '48hr':1},
                            patient_strat= True,
                            ignore=[])

elif args.task == 'HEROHE_enough_3cat':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/HEROHE_enough-sample.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, '3+':2, '2+AMP':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'HEROHE_enough_2cat':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/HEROHE_enough-sample.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':0, '3+':1, '2+AMP':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
                            
elif args.task == 'HEROHE_enough_0vnot0':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/HEROHE_enough-sample.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, '3+':1, '2+AMP':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'All-HER2-HE_3cat':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':2, '3+':2, '2+AMP':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'All-HER2-HE_PosvnotPos':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':0, 'Positive':1, '3+':1, '2+AMP':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'All-HER2-HE_LowvnotLow':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':0, '3+':0, '2+AMP':0},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
                            
elif args.task == 'All-HER2-HE_0vnot0':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/All-HER2-HE-exclude.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'0':0, 'Low':1, 'Positive':1, '3+':1, '2+AMP':1},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
                            
elif args.task == 'Yale-HSHER2_4cat':
    args.n_classes=4
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/489,499 NoNA HS-HER2 CLAM.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Neg':0, 'Low':1, 'Mod':2, 'High':3},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
elif args.task == 'Yale-HSHER2_3cat':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/489,499 NoNA HS-HER2 CLAM.csv',
                            shuffle = True, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'Neg':0, 'Low':1, 'Mod':1, 'High':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])
                            
else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    val_per = int(args.val_frac * 100)
    test_per = int(args.test_frac * 100)
    train_per = int((1-args.val_frac-args.test_frac) * 100)
    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_LF{}_Split{},{},{}'.format(int(lf * 100), train_per, val_per, test_per)
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))




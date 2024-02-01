import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_SB, CLAM_MB
import pdb
import os
import pandas as pd
from utils.utils import *
from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from numpy import interp

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {"dropout": args.drop_out, 'n_classes': args.n_classes}
    
    if args.model_size is not None and args.model_type in ['clam_sb', 'clam_mb']:
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type =='clam_sb':
        model = CLAM_SB(**model_dict)
    elif args.model_type =='clam_mb':
        model = CLAM_MB(**model_dict)
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)

    print_network(model)

    ckpt = torch.load(ckpt_path)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    model.load_state_dict(ckpt_clean, strict=True)

    model.relocate()
    model.eval()
    return model

def eval(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc_score, df, acc_logger
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df
    
def eval_allPerformance(dataset, args, ckpt_path):
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    loader = get_simple_loader(dataset)
    patient_results, test_error, auc, df, roc_df, _ = allPerformance(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df, roc_df

def allPerformance(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)
    
    roc_aucs = {}
    all_fpr = {}
    all_tpr = {}
    all_thresholds = {}
    
    if len(np.unique(all_labels)) == 1:
        auc_score = -1
    else: 
      if args.n_classes == 2:
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:,1])
        auc_score = auc(fpr, tpr)
        all_fpr[0] = fpr
        all_tpr[0] = tpr
        all_thresholds[0] = thresholds
        roc_aucs[0] = auc_score
        # rodData.append([fpr, tpr, thresholds])
        # auc_score = roc_auc_score(all_labels, all_probs[:, 1])
      else:
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        for class_idx in range(args.n_classes):
            if class_idx in all_labels:
                fpr, tpr, thresholds = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                all_fpr[class_idx] = fpr
                all_tpr[class_idx] = tpr
                all_thresholds[class_idx] = thresholds
                roc_aucs[class_idx] = auc(fpr, tpr)
                # rocData.append([fpr, tpr, thresholds])
                # rocData[class_idx] = [fpr, tpr, thresholds]
            else:
                all_fpr[class_idx] = float('nan')
                all_tpr[class_idx] = float('nan')
                all_thresholds[class_idx] = float('nan')
                roc_aucs[class_idx] = float('nan')
                # aucs.append(float('nan'))
        
        #Calculate micro-average ROC
        binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
        fpr, tpr, thresholds = roc_curve(binary_labels.ravel(), all_probs.ravel())
        auc_score = auc(fpr, tpr)
        roc_aucs['micro'] = auc(fpr, tpr)
        all_fpr['micro'] = fpr
        all_tpr['micro'] = tpr
        all_thresholds['micro'] = thresholds
        # rocData['micro_average'] = [fpr, tpr, thresholds]
        # rocData.append([fpr, tpr, thresholds])
        
        #Calculate macro-average ROC
        # First aggregate all false positive rates
        unified_fpr = np.unique(np.concatenate([all_fpr[i] for i in range(args.n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(unified_fpr)
        for i in range(args.n_classes):
          mean_tpr += interp(unified_fpr, all_fpr[i], all_tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= args.n_classes
        
        all_fpr["macro"] = unified_fpr
        all_tpr["macro"] = mean_tpr
        roc_aucs["macro"] = auc(all_fpr["macro"], all_tpr["macro"])
        auc_score = roc_aucs["macro"]
    
    
    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)

    roc_dict = {}
    if args.n_classes > 2:
        for c in range(args.n_classes):
            roc_dict.update({'FPR_class{}'.format(c): all_fpr[c]})
            roc_dict.update({'TPR_class{}'.format(c): all_tpr[c]})
            roc_dict.update({'Thresholds_class{}'.format(c): all_thresholds[c]})
            roc_dict.update({'FPR_micro': all_fpr['micro']})
            roc_dict.update({'TPR_micro': all_tpr['micro']})
            roc_dict.update({'Thresholds_micro': all_thresholds['micro']})
            roc_dict.update({'FPR_macro': all_fpr['macro']})
            roc_dict.update({'TPR_macro': all_tpr['macro']})
    else:
        roc_dict.update({'FPR'.format(c): all_fpr[0]})
        roc_dict.update({'TPR'.format(c): all_tpr[0]})
        roc_dict.update({'Thresholds'.format(c): all_thresholds[0]})
    roc_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in roc_dict.items() ]))
    
    return patient_results, test_error, auc_score, df, roc_df, acc_logger

def summary(model, loader, args):
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)
        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    return patient_results, test_error, auc_score, df, acc_logger

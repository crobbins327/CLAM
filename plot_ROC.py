#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 20:04:37 2022

@author: cjr66
"""
if __name__ == "__main__":
    rocs = []
    all_auc = []
    all_acc = []
    for ckpt_idx in range(len(ckpt_paths)):
        #open roc_df
        #open metrics df
        all_auc.append(auc)
        all_acc.append(1-test_error)
        if args.n_classes > 2:
            rocs.append(roc_df.loc[~pd.isna(roc_df['FPR_macro']), ['FPR_macro', 'TPR_macro']])
        else:
            rocs.append(roc_df.loc[~pd.isna(roc_df['FPR']), ['FPR', 'TPR']])
        # df.to_csv(os.path.join(args.save_dir, 'fold_{}.csv'.format(folds[ckpt_idx])), index=False)
        # roc_df.to_csv(os.path.join(args.save_dir, 'fold_{}_ROC.csv'.format(folds[ckpt_idx])), index=False)
    
    
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
    if len(chkpt_paths) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(folds[0], folds[-1])
    else:
        save_name = 'summary.csv'
    final_df.to_csv(os.path.join(args.save_dir, save_name))

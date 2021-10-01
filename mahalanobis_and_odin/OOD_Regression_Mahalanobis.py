"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV

parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', default="resnet_50", help='resnet | densenet')
args = parser.parse_args()
print(args)

def main():
    # initial setup
    dataset_list = ['skin_cancer']
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005', 'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    
    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = './output/' + args.net_type + '_' + dataset + '/'
        out_list = ['imgnet', 'skin_cli', 'skin_derm', 'corrupted', 'corrupted_70', 'nct']
        
        list_best_results_out, list_best_results_index_out = [[] for i in range(len(out_list))], [[] for i in range(len(out_list))]
        for out in out_list:
            best_lr = None
            best_score = None
            best_tnr = 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_in = total_X[-2533:,:]
                Y_in = total_Y[-2533:]
                X_out = total_X[:-2533,:]
                Y_out = total_Y[:-2533]
                
                l = int(0.8*X_out.shape[0])
                
                X_train = np.concatenate((X_in[:500], X_out[:l]))
                Y_train = np.concatenate((Y_in[:500], Y_out[:l]))
                
                X_val_for_test = np.concatenate((X_in[500:550], X_out[l:2*l]))
                Y_val_for_test = np.concatenate((Y_in[500:550], Y_out[l:2*l]))
                
                lr = LogisticRegressionCV(n_jobs=-1,max_iter=1000).fit(X_train, Y_train)
                results = lib_regression.detection_performance(lr, X_val_for_test, Y_val_for_test, outf)
                
                if best_tnr < results['TMP']['TNR']:
                    best_tnr = results['TMP']['TNR']
                    best_lr = lr
                    best_score = score 

            for i,out in enumerate(out_list):
                print('Out-of-distribution: ', out)
                total_X, total_Y = lib_regression.load_characteristics(best_score, dataset, out, outf)
                X_in = total_X[-2533:,:]
                Y_in = total_Y[-2533:]
                X_out = total_X[:-2533,:]
                Y_out = total_Y[:-2533]
                
                np.random.seed(seed=0)
                np.random.shuffle(X_out)

                k = int(0.1*X_out.shape[0])
                l = int(0*X_out.shape[0])
                
                # X_train = np.concatenate((X_in[:k], X_out[:l]))
                # Y_train = np.concatenate((Y_in[:k], Y_out[:l]))
                
                # X_val_for_test = np.concatenate((X_in[k:2*k], X_out[l:2*l]))
                # Y_val_for_test = np.concatenate((Y_in[k:2*k], Y_out[l:2*l]))
                
                X_test = np.concatenate((X_in[550:], X_out[2*l:]))
                Y_test = np.concatenate((Y_in[550:], Y_out[2*l:]))
                    
                best_result = lib_regression.detection_performance(best_lr, X_test, Y_test, outf)
                list_best_results_out[i].append(best_result)
                list_best_results_index_out[i].append(best_score)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)
    
    print(len(list_best_results_out))
    print(len(list_best_results))
    # print the results
    count_in = 0
    mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    
    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        # out_list = ['skin_cli', 'skin_derm', 'corrupted', 'corrupted_70', 'imgnet', 'nct', 'final_test']
        print(len(in_list))
        count_out = 0
        for results in in_list:
            print('out_distribution: '+ out_list[count_out])
            summary_results = {"TMP":{}}
            
            for r in results:
                # print(r)
                summary_results["TMP"]["TNR"] = summary_results["TMP"].get("TNR",0)+r["TMP"]["TNR"]/len(out_list)
                summary_results["TMP"]["AUROC"] = summary_results["TMP"].get("AUROC",0)+r["TMP"]["AUROC"]/len(out_list)
                summary_results["TMP"]["DTACC"] = summary_results["TMP"].get("DTACC",0)+r["TMP"]["DTACC"]/len(out_list)
                summary_results["TMP"]["AUIN"] = summary_results["TMP"].get("AUIN",0)+r["TMP"]["AUIN"]/len(out_list)
                summary_results["TMP"]["AUOUT"] = summary_results["TMP"].get("AUOUT",0)+r["TMP"]["AUOUT"]/len(out_list)
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            print('\n{val:6.2f}'.format(val=100.*summary_results['TMP']['TNR']), end='')
            print(' {val:6.2f}'.format(val=100.*summary_results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(val=100.*summary_results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(val=100.*summary_results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(val=100.*summary_results['TMP']['AUOUT']), end='')
            print('Input noise: ' + str(list_best_results_index[count_in][count_out]))
            print('')
            count_out += 1
        count_in += 1

if __name__ == '__main__':
    main()
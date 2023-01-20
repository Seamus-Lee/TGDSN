import torch
import numpy as np

import pandas as pd
from models import *
from TGDSN import *
import os
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# def main():


parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
cuda = True
print(torch.cuda.is_available())
print(torch.__version__)

if __name__ == '__main__':
    select_method = 'TGAT'

    hyper_param = {'FD001_FD002': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.95},
                   'FD001_FD003': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.9},
                   'FD001_FD004': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD002_FD001': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD002_FD003': {'epochs': 100, 'batch_size': 256, 'lr': 5e-5, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.005,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.95},
                   'FD002_FD004': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.97},
                   'FD003_FD001': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD003_FD002': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD003_FD004': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD004_FD001': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.005,
                                   'lambda_f': 0.001, 'StepLR': False, 'gamma': 0.99},
                   'FD004_FD002': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.05,
                                   'lambda_f': 0.01, 'StepLR': False, 'gamma': 0.99},
                   'FD004_FD003': {'epochs': 100, 'batch_size': 256, 'lr': 5e-3, 'lambda_rul': 0.1, 'lambda_d': 10,
                                   'lambda_r': 0.005,
                                   'lambda_f': 0.001, 'StepLR': False, 'gamma': 0.99}}

    data_path = "D:/TGDSN/data/dataset.pt"
    my_dataset = torch.load(data_path)
    config = {"model_name": 'TGAT', "num_node_features": 14}

    config.update(
        {'num_runs': 1, 'save': False, 'tensorboard': False, 'tsne': True, 'tensorboard_epoch': False, 'k_disc': 100,
         'k_clf': 1, 'iterations': 1, 'pic': False})

    df = pd.DataFrame();
    res = [];
    full_res = []
    print('=' * 89)
    print(f'Domain Adaptation using: {select_method}')
    print('=' * 89)
    for src_id in ['FD001']:  # 'FD001', 'FD002', 'FD003', 'FD004'
        for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:  # 'FD001', 'FD002', 'FD003', 'FD004'
            if src_id != tgt_id:
                total_loss = []
                total_score = []
                for run_id in range(config['num_runs']):
                    test_loss, test_score, best_loss, best_epoch, best_loss_score, best_score, best_score_epoch, manual_seed = TGDSN(
                        hyper_param, device,
                        config, TGAT,
                        my_dataset, src_id,
                        tgt_id, args)
                    total_loss.append(test_loss)
                    total_score.append(test_score)
                loss_mean, loss_std = np.mean(np.array(total_loss)), np.std(np.array(total_loss))
                score_mean, score_std = np.mean(np.array(total_score)), np.std(np.array(total_score))
                full_res.append((f'run_id:{run_id}', f'{src_id}-->{tgt_id}', f'{loss_mean:2.2f}', f'{score_mean:2.2f}',
                                 f'{best_loss:2.2f}', f'{best_epoch:2.2f}', f'{best_loss_score:2.2f}', f'{best_score:2.2f}',
                                 f'{best_score_epoch:2.2f}', f'{manual_seed:2.2f}'))

    df = df.append(pd.Series((f'{select_method}')), ignore_index=True)

    df = df.append(pd.Series(
        ("run_id", 'scenario', 'mean_loss', 'mean_score', 'best_loss', 'best_epoch', 'best_loss_score', 'best_score',
         'best_score_epoch', 'manual_seed')),
        ignore_index=True)

    df = df.append(pd.DataFrame(full_res), ignore_index=True)
    print('=' * 89)
    print(f'Results using: {select_method}')
    print('=' * 89)
    print(df.to_string())


"""
Standalone evaluation script for running final test set evaluation on a separate machine.
This script loads the exported datasets and runs the same evaluation as the main training script.

Usage:
    python run_separate_evaluation.py --file-name openml_616 --task cls
"""

import os
import sys
import argparse
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append('./')

from logger import *
from task_mapping import task_dict, task_type, base_path
from tools import test_task_separate, test_task_separate_ag


def init_param():
    parser = argparse.ArgumentParser(description='Separate Test Set Evaluation')
    parser.add_argument('--file-name', type=str, required=True,
                        help='data name (must match the exported dataset prefix)')
    parser.add_argument('--task', type=str, default='ng',
                        help='ng/cls/reg/det (if ng, will use task_dict mapping)')
    parser.add_argument('--log-level', type=str, default='info',
                        help='log level')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed')
    args = parser.parse_args()
    return args


def run_evaluation(file_name, task_name, seed):
    """Run evaluation on the exported datasets."""

    # Load datasets
    info('Loading exported datasets...')
    original_train_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_original_train.csv'
    original_val_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_original_val.csv'
    original_test_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_original_test.csv'
    generated_train_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_generated_train.csv'
    generated_val_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_generated_val.csv'
    generated_test_path = base_path + f'{file_name}/seed{seed}/' + file_name + '_eval_generated_test.csv'

    # Check if all files exist
    required_files = [
        original_train_path, original_val_path, original_test_path,
        generated_train_path, generated_val_path, generated_test_path
    ]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        error(f'Missing required dataset files:')
        for f in missing_files:
            error(f'  - {f}')
        error('Please ensure all datasets have been exported before running evaluation.')
        sys.exit(1)

    # Load datasets
    D_original_train = pd.read_csv(original_train_path)
    D_original_val = pd.read_csv(original_val_path)
    D_original_test = pd.read_csv(original_test_path)
    D_OPT_train = pd.read_csv(generated_train_path)
    D_OPT_val = pd.read_csv(generated_val_path)
    D_OPT_test = pd.read_csv(generated_test_path)

    info(f'Loaded datasets from {base_path}')
    info(f'  Original train: {D_original_train.shape}')
    info(f'  Original val: {D_original_val.shape}')
    info(f'  Original test: {D_original_test.shape}')
    info(f'  Generated train: {D_OPT_train.shape}')
    info(f'  Generated val: {D_OPT_val.shape}')
    info(f'  Generated test: {D_OPT_test.shape}')

    # Run evaluation based on task type
    info('========== Final Evaluation on Separate Test Set ==========')

    if task_name == 'reg':
        mae0_test, mse0_test, rmse0_test = test_task_separate(
            D_original_train, D_original_test, task=task_name)
        mae1_test, mse1_test, rmse1_test = test_task_separate(
            D_OPT_train, D_OPT_test, task=task_name)

        # Add TabularPredictor evaluation
        info('========== TabularPredictor Evaluation on Separate Test Set ==========')
        mae0_test_ag, mse0_test_ag, rmse0_test_ag = test_task_separate_ag(
            D_original_train, D_original_val, D_original_test, task=task_name)
        mae1_test_ag, mse1_test_ag, rmse1_test_ag = test_task_separate_ag(
            D_OPT_train, D_OPT_val, D_OPT_test, task=task_name)

        info('[SEPARATE TEST] 1-MAE on original is: {:.5f}, 1-MAE on generated is: {:.5f}'.
             format(mae0_test, mae1_test))
        info('[SEPARATE TEST] 1-MSE on original is: {:.5f}, 1-MSE on generated is: {:.5f}'.
             format(mse0_test, mse1_test))
        info('[SEPARATE TEST] 1-RMSE on original is: {:.5f}, 1-RMSE on generated is: {:.5f}'.
             format(rmse0_test, rmse1_test))
        info('[SEPARATE TEST AG] 1-RMSE on original is: {:.5f}, 1-RMSE on generated is: {:.5f}'.
             format(rmse0_test_ag if rmse0_test_ag is not None else 0.0,
                    rmse1_test_ag if rmse1_test_ag is not None else 0.0))

        # Return results
        return {
            'task': 'regression',
            'original': {
                'mae': mae0_test, 'mse': mse0_test, 'rmse': rmse0_test,
                'rmse_ag': rmse0_test_ag
            },
            'generated': {
                'mae': mae1_test, 'mse': mse1_test, 'rmse': rmse1_test,
                'rmse_ag': rmse1_test_ag
            }
        }

    elif task_name == 'cls':
        acc0_test, precision0_test, recall0_test, f1_0_test = test_task_separate(
            D_original_train, D_original_test, task=task_name)
        acc1_test, precision1_test, recall1_test, f1_1_test = test_task_separate(
            D_OPT_train, D_OPT_test, task=task_name)

        # Add TabularPredictor evaluation
        info('========== TabularPredictor Evaluation on Separate Test Set ==========')
        acc0_test_ag, precision0_test_ag, recall0_test_ag, f1_0_test_ag = test_task_separate_ag(
            D_original_train, D_original_val, D_original_test, task=task_name)
        acc1_test_ag, precision1_test_ag, recall1_test_ag, f1_1_test_ag = test_task_separate_ag(
            D_OPT_train, D_OPT_val, D_OPT_test, task=task_name)
        info('[SEPARATE TEST] Acc on original is: {:.5f}, Acc on generated is: {:.5f}'.
             format(acc0_test, acc1_test))
        info('[SEPARATE TEST AG] Acc on original is: {:.5f}, Acc on generated is: {:.5f}'.
             format(acc0_test_ag if acc0_test_ag is not None else 0.0,
                    acc1_test_ag if acc1_test_ag is not None else 0.0))

        # Return results
        return {
            'task': 'classification',
            'original': {
                'acc': acc0_test, 'precision': precision0_test,
                'recall': recall0_test, 'f1': f1_0_test,
                'acc_ag': acc0_test_ag
            },
            'generated': {
                'acc': acc1_test, 'precision': precision1_test,
                'recall': recall1_test, 'f1': f1_1_test,
                'acc_ag': acc1_test_ag
            }
        }

    elif task_name == 'det':
        map0_test, f1_0_test, ras0_test = test_task_separate(
            D_original_train, D_original_test, task=task_name)
        map1_test, f1_1_test, ras1_test = test_task_separate(
            D_OPT_train, D_OPT_test, task=task_name)

        info('[SEPARATE TEST] Average Precision Score on original is: {:.5f}, '
             'Average Precision Score on generated is: {:.5f}'.format(map0_test, map1_test))
        info('[SEPARATE TEST] F1 Score on original is: {:.5f}, F1 Score on generated is: {:.5f}'.
             format(f1_0_test, f1_1_test))
        info('[SEPARATE TEST] ROC AUC Score on original is: {:.5f}, ROC AUC Score on generated is: {:.5f}'.
             format(ras0_test, ras1_test))

        # Return results
        return {
            'task': 'detection',
            'original': {'map': map0_test, 'f1': f1_0_test, 'roc_auc': ras0_test},
            'generated': {'map': map1_test, 'f1': f1_1_test, 'roc_auc': ras1_test}
        }
    else:
        error(f'Unknown task type: {task_name}')
        sys.exit(1)


if __name__ == '__main__':
    args = init_param()

    # Determine task name
    if args.task == 'ng':
        if args.file_name in task_dict:
            task_name = task_dict[args.file_name]
        else:
            error(f'File name {args.file_name} not found in task_dict. Please specify --task explicitly.')
            sys.exit(1)
    else:
        if args.task not in task_type:
            error(f'Invalid task type: {args.task}. Must be one of {task_type}')
            sys.exit(1)
        task_name = args.task

    info(f'Running evaluation for dataset: {args.file_name}')
    info(f'Task type: {task_name}')

    results = run_evaluation(args.file_name, task_name, args.seed)

    info('========== Evaluation Complete ==========')
    info(f'Results summary: {results}')

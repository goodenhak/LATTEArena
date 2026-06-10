"""
baseline.py - Evaluate baseline performance using AutoGluon on original features.

This script evaluates the initial model performance before any feature engineering,
providing a baseline for comparison with feature engineering methods.

Usage:
    python baseline.py --data_name credit-g --seed 1
    python baseline.py --data_name credit-g --seeds 1 2 3 4 5 6
    python baseline.py --datasets credit-g vehicle --seeds 1 2 3
"""

import argparse
import os
import time
import json
import logging
from datetime import datetime

import Preprocess
import Evaluator
import utils


# ============================================================
# Default Configurations
# ============================================================

ALL_DATASETS = [
    'credit-g',
    'credit-approval',
    'kc1',
    'qsar-biodeg',
    'vehicle',
    'heart-h',
    'electricity',
    'balance-scale',
]


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Evaluate baseline performance using AutoGluon',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--data_name', type=str, default=None,
                        help='Single dataset name')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Multiple dataset names')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single random seed')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Multiple random seeds')

    parser.add_argument('--task_type', type=int, default=1,
                        help='Task type: 1=classification, 0=regression')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size ratio')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation set size ratio')

    parser.add_argument('--downstream', type=str, default='both',
                        choices=['rf', 'ag', 'both'],
                        help='Downstream model: rf=RandomForest, ag=AutoGluon, both=both')

    parser.add_argument('--log_path', type=str, default='./log',
                        help='Log directory')
    parser.add_argument('--output_json', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')

    return parser.parse_args()


# ============================================================
# Logging Setup
# ============================================================

def setup_logging(log_path: str, log_filename: str):
    """Setup logging configuration."""
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, log_filename)

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Also print to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(console_handler)

    return logger


# ============================================================
# Baseline Evaluation
# ============================================================

def evaluate_baseline(data_name: str, seed: int, task_type: int,
                      test_size: float, val_size: float,
                      downstream: str, verbose: bool = False) -> dict:
    """
    Evaluate baseline performance on original features.

    Returns:
        dict with keys: data_name, seed, val_acc_rf, test_acc_rf,
                        val_acc_ag, test_acc_ag, num_features, num_samples
    """
    result = {
        'data_name': data_name,
        'seed': seed,
        'task_type': task_type,
        'val_acc_rf': None,
        'test_acc_rf': None,
        'val_acc_ag': None,
        'test_acc_ag': None,
        'num_features': None,
        'num_train_samples': None,
        'num_val_samples': None,
        'num_test_samples': None,
    }

    try:
        # Prepare data
        utils.copy_and_rename_metadata(data_name)

        df, df_train, df_test, df_val, target, label_list = Preprocess.split_and_prepare_data(
            data_name=data_name,
            task_type=task_type,
            seed=seed,
            test_size=test_size,
            val_size=val_size
        )

        # Record data info
        result['num_features'] = df_train.shape[1] - 1  # exclude target
        result['num_train_samples'] = len(df_train)
        result['num_val_samples'] = len(df_val)
        result['num_test_samples'] = len(df_test)

        # Load dataset
        train_data, val_data, test_data = Evaluator.load_dataset(data_name)

        # Evaluate with RandomForest
        if downstream in ['rf', 'both']:
            _, val_acc_rf = Evaluator.train_and_evaluate_rf(
                train_data, val_data, target, task_type)
            _, test_acc_rf = Evaluator.train_and_evaluate_rf(
                train_data, test_data, target, task_type)
            result['val_acc_rf'] = val_acc_rf
            result['test_acc_rf'] = test_acc_rf

        # Evaluate with AutoGluon
        if downstream in ['ag', 'both']:
            _, val_acc_ag = Evaluator.train_and_evaluate(
                train_data, val_data, target, task_type)
            _, test_acc_ag = Evaluator.train_and_evaluate(
                train_data, test_data, target, task_type)
            result['val_acc_ag'] = val_acc_ag
            result['test_acc_ag'] = test_acc_ag

        result['status'] = 'success'

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)

    return result


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    # Determine datasets
    if args.datasets:
        datasets = args.datasets
    elif args.data_name:
        datasets = [args.data_name]
    else:
        datasets = ALL_DATASETS

    # Determine seeds
    if args.seeds:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [1, 2, 3, 4, 5, 6]

    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"baseline_{timestamp}.log"
    logger = setup_logging(args.log_path, log_filename)

    logger.info("=" * 70)
    logger.info("Baseline Performance Evaluation")
    logger.info("=" * 70)
    logger.info(f"Datasets: {datasets}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Downstream: {args.downstream}")
    logger.info(f"Task type: {args.task_type}")
    logger.info("=" * 70)

    all_results = []
    start_time = time.time()

    total = len(datasets) * len(seeds)
    current = 0

    for data_name in datasets:
        for seed in seeds:
            current += 1
            logger.info(f"\n[{current}/{total}] Evaluating: {data_name} | seed={seed}")

            result = evaluate_baseline(
                data_name=data_name,
                seed=seed,
                task_type=args.task_type,
                test_size=args.test_size,
                val_size=args.val_size,
                downstream=args.downstream,
                verbose=args.verbose
            )

            all_results.append(result)

            if result['status'] == 'success':
                msg = f"    Features: {result['num_features']}, Samples: {result['num_train_samples']}"
                if result['val_acc_rf'] is not None:
                    msg += f"\n    RF  - Val: {result['val_acc_rf']:.4f}, Test: {result['test_acc_rf']:.4f}"
                if result['val_acc_ag'] is not None:
                    msg += f"\n    AG  - Val: {result['val_acc_ag']:.4f}, Test: {result['test_acc_ag']:.4f}"
                logger.info(msg)
            else:
                logger.error(f"    FAILED: {result.get('error', 'Unknown error')}")

    total_time = time.time() - start_time

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total time: {total_time:.1f}s")

    # Group by dataset
    for data_name in datasets:
        dataset_results = [r for r in all_results if r['data_name'] == data_name and r['status'] == 'success']
        if not dataset_results:
            continue

        logger.info(f"\n{data_name}:")

        if dataset_results[0]['val_acc_rf'] is not None:
            val_rf = [r['val_acc_rf'] for r in dataset_results]
            test_rf = [r['test_acc_rf'] for r in dataset_results]
            logger.info(f"  RF  - Val: {sum(val_rf)/len(val_rf):.4f} ± {max(val_rf)-min(val_rf):.4f}, "
                       f"Test: {sum(test_rf)/len(test_rf):.4f} ± {max(test_rf)-min(test_rf):.4f}")

        if dataset_results[0]['val_acc_ag'] is not None:
            val_ag = [r['val_acc_ag'] for r in dataset_results]
            test_ag = [r['test_acc_ag'] for r in dataset_results]
            logger.info(f"  AG  - Val: {sum(val_ag)/len(val_ag):.4f} ± {max(val_ag)-min(val_ag):.4f}, "
                       f"Test: {sum(test_ag)/len(test_ag):.4f} ± {max(test_ag)-min(test_ag):.4f}")

    logger.info("=" * 70)

    # Save results to JSON
    output_json = args.output_json or os.path.join(args.log_path, f"baseline_results_{timestamp}.json")
    with open(output_json, 'w') as f:
        json.dump({
            'config': {
                'datasets': datasets,
                'seeds': seeds,
                'task_type': args.task_type,
                'downstream': args.downstream,
            },
            'results': all_results,
            'total_time': total_time
        }, f, indent=2)

    logger.info(f"\nResults saved to: {output_json}")

    # Print table format for easy copy
    print("\n" + "=" * 70)
    print("TABLE FORMAT (for paper/report)")
    print("=" * 70)
    print(f"{'Dataset':<20} {'RF Val':<10} {'RF Test':<10} {'AG Val':<10} {'AG Test':<10}")
    print("-" * 70)

    for data_name in datasets:
        dataset_results = [r for r in all_results if r['data_name'] == data_name and r['status'] == 'success']
        if not dataset_results:
            continue

        rf_val = rf_test = ag_val = ag_test = "-"

        if dataset_results[0]['val_acc_rf'] is not None:
            rf_val = f"{sum(r['val_acc_rf'] for r in dataset_results)/len(dataset_results):.4f}"
            rf_test = f"{sum(r['test_acc_rf'] for r in dataset_results)/len(dataset_results):.4f}"

        if dataset_results[0]['val_acc_ag'] is not None:
            ag_val = f"{sum(r['val_acc_ag'] for r in dataset_results)/len(dataset_results):.4f}"
            ag_test = f"{sum(r['test_acc_ag'] for r in dataset_results)/len(dataset_results):.4f}"

        print(f"{data_name:<20} {rf_val:<10} {rf_test:<10} {ag_val:<10} {ag_test:<10}")

    print("=" * 70)


if __name__ == "__main__":
    main()

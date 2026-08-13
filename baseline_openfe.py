"""OpenFE baseline under the LATTEArena evaluation protocol.

Fair comparison with the other methods in this framework:
- Same data source and preprocessing: Preprocess.split_and_prepare_data
  (reads tabular_data/{data_name}.csv, label-encodes the target, stratified
  80/20 test split, then 80/20 train/val split, random_state=seed) — identical
  to what latte.py uses for every method.
- Same downstream evaluator: Evaluator.train_and_evaluate_rf
  (RandomForestClassifier(random_state=42), same label-encoding / median-fill
  / scaling), for both the initial (original features) and final
  (OpenFE features) accuracies.

OpenFE itself (feature search) is the reproduce-repo implementation
(OpenFE_reproduce/OpenFE.py) with the paper's defaults: first-order candidate
enumeration, LightGBM-based successive featurewise selection, 10 saved
features. OpenFE's internal LightGBM scoring is part of the method; the final
reported numbers always come from the framework's RF evaluator.

Log format matches extract.py conventions:
    INFO - val_acc = ... / INFO - test_acc = ...        (initial, original features)
    INFO - Best performance = ...                        (val, OpenFE features)
    INFO - final_test_acc = ...                          (test, OpenFE features)

Usage:
    python baseline_openfe.py --data_name credit-g --seed 1
    python baseline_openfe.py --data_name credit-g --seed 1 \
        --log_path execution_logs/Main_Results/OpenFE
"""

import os
import re
import sys
import time
import argparse
import logging
import warnings

import numpy as np
import pandas as pd

import Preprocess
import Evaluator

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenFE_reproduce"))
from OpenFE import OpenFE, get_candidate_features  # noqa: E402

# --- Compatibility patch (does not modify the OpenFE repo) ---
# OpenFE's stage-1 successive halving starts from tiny subsample blocks
# (fold=32); on small datasets the first val block may contain a single
# class, and sklearn's log_loss then requires explicit labels. OpenFE
# assumes large datasets, so we patch get_init_metric to pass labels.
import scipy.special as _scipy_special  # noqa: E402
from sklearn.metrics import log_loss as _sk_log_loss  # noqa: E402

_orig_get_init_metric = OpenFE.get_init_metric


def _get_init_metric_safe(self, pred, label):
    try:
        return _orig_get_init_metric(self, pred, label)
    except ValueError:
        if self.metric == 'binary_logloss':
            return _sk_log_loss(label, _scipy_special.expit(pred), labels=[0, 1])
        if self.metric == 'multi_logloss':
            n_classes = pred.shape[1] if np.ndim(pred) > 1 else 2
            return _sk_log_loss(label, _scipy_special.softmax(pred, axis=1),
                                labels=list(range(n_classes)))
        raise


OpenFE.get_init_metric = _get_init_metric_safe

# delete_same() collapses consecutive equal scores; on tiny first blocks all
# candidates can tie and the list shrinks to 1, crashing a debug print in
# stage1_select (line 319). Keep at least 2 entries in that degenerate case.
_orig_delete_same = OpenFE.delete_same


def _delete_same_safe(self, candidate_features_scores, threshold=1e-20):
    backup = list(candidate_features_scores)
    out = _orig_delete_same(self, candidate_features_scores, threshold)
    if len(out) < 2 and len(backup) >= 2:
        return backup[:2]
    return out


OpenFE.delete_same = _delete_same_safe

# stage2_select() crashes on np.vstack when stage 1 selected no features at
# all (happens on some splits of small datasets). Return an empty list
# instead; main() then falls back to the original features.
_orig_stage2_select = OpenFE.stage2_select


def _stage2_select_safe(self):
    if not self.candidate_features_list:
        return []
    return _orig_stage2_select(self)


OpenFE.stage2_select = _stage2_select_safe

warnings.filterwarnings("ignore")


def parse_arguments():
    parser = argparse.ArgumentParser(description='OpenFE Feature Engineering Baseline')

    parser.add_argument('--log_path', type=str, default='./log',
                        help='Path to log directory')
    parser.add_argument('--log_filename', type=str, default=None,
                        help='Log filename (default: {data_name}_OpenFE_{seed}.log)')

    parser.add_argument('--data_name', type=str, default='credit-g',
                        help='Dataset name (must exist in tabular_data/)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed (also selects the train/val/test split)')
    parser.add_argument('--task_type', type=int, default=1,
                        help='1 = classification, 0 = regression')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

    # OpenFE parameters (defaults follow the reproduce repo's runs/FE_first_order.py)
    parser.add_argument('--n_new_features', type=int, default=10,
                        help='Number of OpenFE features to keep (n_saved_features)')
    parser.add_argument('--ordinal_threshold', type=int, default=100)
    parser.add_argument('--remain', type=int, default=2000,
                        help='Candidates kept after stage 1')
    parser.add_argument('--fold', type=int, default=32)
    parser.add_argument('--n_jobs', type=int, default=8)

    return parser.parse_args()


def load_split(data_name, task_type, seed, test_size, val_size):
    """Load data and reproduce the framework's canonical split.

    Primary path: Preprocess.split_and_prepare_data (reads
    tabular_data/{data_name}.csv). Fallback: tmp/{data_name}/{data_name}.csv,
    the full target-encoded dataframe cached by a previous run — the split is
    re-done with the exact same logic (convert_dtypes + stratified
    train_test_split with random_state=seed), so the resulting
    train/val/test rows are identical to the framework's.
    """
    if os.path.exists(os.path.join('tabular_data', data_name + '.csv')):
        return Preprocess.split_and_prepare_data(
            data_name=data_name, task_type=task_type, seed=seed,
            test_size=test_size, val_size=val_size)

    from sklearn.model_selection import train_test_split
    df = pd.read_csv(os.path.join('tmp', data_name, data_name + '.csv'))
    target = 'target'
    X = df.convert_dtypes()
    y = df[target].to_numpy()
    X = X.drop(target, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
        stratify=y if task_type == 1 else None)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=seed,
        stratify=y_train if task_type == 1 else None)
    label_list = np.unique(y).tolist() if task_type == 1 else None
    df_train = X_train.copy(); df_train[target] = y_train
    df_val = X_val.copy(); df_val[target] = y_val
    df_test = X_test.copy(); df_test[target] = y_test
    return df, df_train, df_test, df_val, target, label_list


def setup_logging(log_path, log_filename):
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()
    os.makedirs(log_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_path, log_filename),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logger


def main():
    args = parse_arguments()

    # OpenFE_reproduce's parallel workers rely on fork-inherited module
    # globals (_data/_label/_init_scores); macOS defaults to 'spawn'.
    import multiprocessing as mp
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass
    task_type = args.task_type

    if args.log_filename is None:
        args.log_filename = f"{args.data_name}_OpenFE_{args.seed}.log"
    logger = setup_logging(args.log_path, args.log_filename)
    logger.info("========== START ==========")
    logger.info(f"Arguments: {vars(args)}")
    total_start_time = time.time()

    # --- Framework data + split + preprocessing (identical to latte.py) ---
    df, df_train, df_test, df_val, target, label_list = load_split(
        data_name=args.data_name,
        task_type=task_type,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    # convert_dtypes() yields pandas 'string' dtype; the framework's methods
    # re-read the split CSVs and therefore see 'object'. Match that, because
    # Evaluator.train_and_evaluate_rf branches on dtype == 'object'.
    # Nullable Int64/Float64 are likewise normalized to float64: the CSV
    # round-trip produces float64, and Evaluator's median fillna would
    # otherwise fail on Int64 columns with NA.
    for _df in (df_train, df_val, df_test):
        for _col in _df.columns:
            if str(_df[_col].dtype) == "string":
                _df[_col] = _df[_col].astype(object)
            elif str(_df[_col].dtype) in ("Int64", "Float64", "boolean"):
                _df[_col] = _df[_col].astype("float64")
    logger.info(f"train={df_train.shape}, val={df_val.shape}, test={df_test.shape}")

    # --- Initial performance with the framework RF evaluator ---
    _, init_val = Evaluator.train_and_evaluate_rf(df_train, df_val, target, task_type)
    _, init_test = Evaluator.train_and_evaluate_rf(df_train, df_test, target, task_type)
    logger.info(f"val_acc = {init_val}")
    logger.info(f"test_acc = {init_test}")

    # --- Prepare OpenFE input (train+val for search; test only for transform) ---
    X_tv = pd.concat([df_train.drop(columns=[target]), df_val.drop(columns=[target])], axis=0)
    X_tv.index = range(len(X_tv))
    y_tv = pd.concat([df_train[target], df_val[target]], axis=0)
    y_tv.index = range(len(y_tv))
    y_tv = y_tv.to_frame()
    train_index = X_tv[:len(df_train)].index
    val_index = X_tv[len(df_train):].index

    X_test = df_test.drop(columns=[target]).copy()
    X_test.index = range(len(X_test))

    # convert_dtypes() yields nullable Int64/Float64 whose pd.NA breaks
    # OpenFE's float casting; normalize to plain float64 (pd.NA -> np.nan).
    for _X in (X_tv, X_test):
        for _col in _X.columns:
            if str(_X[_col].dtype) in ("Int64", "Float64", "boolean"):
                _X[_col] = _X[_col].astype("float64")

    # Categorical features -> category codes (as in OpenFE's own scripts)
    cat_features = list(X_tv.select_dtypes(exclude=np.number).columns)
    n_tv = len(X_tv)
    _data = pd.concat([X_tv, X_test], axis=0)
    for feature in cat_features:
        _data[feature] = _data[feature].astype('category')
        _data[feature] = _data[feature].cat.codes
        _data[feature] = _data[feature].astype('category')
    X_tv, X_test = _data[:n_tv], _data[n_tv:]

    # OpenFE feeds column names to LightGBM; LightGBM rejects JSON special
    # characters (e.g. '[', ']', '(', ')', '/', '-'). qsar-biodeg contains
    # such names, so sanitize before feature generation.
    def _safe_name(name, seen):
        safe = re.sub(r'[^A-Za-z0-9_]', '_', str(name))
        if not safe or safe[0].isdigit():
            safe = 'f' + safe
        base = safe
        i = 1
        while safe in seen:
            safe = f"{base}_{i}"
            i += 1
        return safe

    original_columns = list(X_tv.columns)
    safe_columns = []
    for c in original_columns:
        safe_columns.append(_safe_name(c, safe_columns))
    column_map = dict(zip(original_columns, safe_columns))
    X_tv.columns = safe_columns
    X_test.columns = safe_columns
    cat_features = [column_map[c] for c in cat_features]

    ord_features, num_features = [], []
    for feature in X_tv.select_dtypes(include=np.number).columns:
        if X_tv[feature].nunique() <= args.ordinal_threshold:
            ord_features.append(feature)
        else:
            num_features.append(feature)
    logger.info(f"categorical={len(cat_features)}, ordinal={len(ord_features)}, numerical={len(num_features)}")

    candidate_features_list = get_candidate_features(
        numerical_features=num_features,
        categorical_features=cat_features,
        ordinal_features=ord_features,
    )
    logger.info(f"candidate features: {len(candidate_features_list)}")

    if task_type == 1:
        n_classes = y_tv[y_tv.columns[0]].nunique()
        metric = 'binary_logloss' if n_classes <= 2 else 'multi_logloss'
        task = 'classification'
        y_ofe = y_tv
    else:
        metric = 'rmse'
        task = 'regression'
        # OpenFE's own scripts standardize regression targets before the
        # search (rmse-based scoring and early stopping are scale-dependent).
        # Only the search sees the standardized y; the framework evaluation
        # below still uses the original target.
        y_ofe = y_tv.copy()
        _col = y_ofe.columns[0]
        _mean = y_ofe[_col][:len(df_train)].mean()
        _std = y_ofe[_col][:len(df_train)].std()
        y_ofe[_col] = (y_ofe[_col] - _mean) / _std

    # --- OpenFE feature search (LightGBM-based, internal to the method) ---
    # fold controls stage-1 successive halving: the first block holds
    # len/fold rows. The paper's fold=32 targets large datasets; on small
    # datasets the first block degenerates (all candidates tie and get
    # dropped). Shrink fold so the first block has ~>=50 rows.
    fold = min(args.fold, max(1, (len(train_index) + len(val_index)) // 50))
    logger.info(f"Starting OpenFE feature search (fold={fold})...")
    ofe = OpenFE()
    ofe.fit(
        data=X_tv, label=y_ofe,
        candidate_features_list=candidate_features_list,
        train_index=train_index, val_index=val_index,
        task=task, metric=metric,
        categorical_features=cat_features,
        remain=args.remain, fold=fold,
        n_jobs=args.n_jobs, seed=args.seed,
    )
    search_time = time.time() - total_start_time
    logger.info(f"OpenFE search finished ({search_time:.2f}s), "
                f"selected {len(ofe.new_features_list)} features")

    # --- Generate the top features for train/val/test ---
    if ofe.new_features_list:
        X_tv_new, X_test_new = ofe.transform(X_tv, X_test, n_new_features=args.n_new_features)
    else:
        # OpenFE selected nothing on this split: evaluate original features
        logger.info("OpenFE selected no features; using original features")
        X_tv_new, X_test_new = X_tv, X_test
    X_train_new = X_tv_new[:len(df_train)]
    X_val_new = X_tv_new[len(df_train):]

    train_data_new = X_train_new.copy()
    train_data_new[target] = df_train[target].values
    val_data_new = X_val_new.copy()
    val_data_new[target] = df_val[target].values
    test_data_new = X_test_new.copy()
    test_data_new[target] = df_test[target].values
    logger.info(f"Original feature count: {df_train.shape[1] - 1}")
    logger.info(f"New feature count: {X_train_new.shape[1]}")

    # --- Final evaluation with the framework RF evaluator ---
    _, val_acc = Evaluator.train_and_evaluate_rf(train_data_new, val_data_new, target, task_type)
    _, test_acc = Evaluator.train_and_evaluate_rf(train_data_new, test_data_new, target, task_type)
    logger.info(f"Best performance = {val_acc}")
    logger.info(f"final_test_acc = {test_acc}")

    # --- AutoGluon evaluation (same protocol as the other methods' logs;
    # extract.py pairs final_test_acc_ag with its AG_INIT reference) ---
    logger.info("Evaluating with AutoGluon...")
    predictor, _ = Evaluator.train_and_evaluate(train_data_new, val_data_new, target, task_type)
    if task_type == 1:
        test_acc_ag = predictor.evaluate(test_data_new)['accuracy']
    else:
        test_acc_ag = predictor.evaluate(test_data_new)['root_mean_squared_error'] + 1
    logger.info(f"final_test_acc_ag = {test_acc_ag}")

    logger.info(f"Total time used = {time.time() - total_start_time:.2f} seconds")
    logger.info("========== END ==========")


if __name__ == "__main__":
    main()

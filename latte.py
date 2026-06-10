"""
latte.py - Unified LLM4FE framework entry point.

Usage:
    python latte.py --method CoT --data_name credit-g ...
    python latte.py --method CoT --data_name credit-g --top 3 ...
    python latte.py --method LtM --data_name credit-g ...
    python latte.py --method Critic --data_name credit-g ...
    python latte.py --method OPRO --data_name credit-g ...
    python latte.py --method OPROc --data_name credit-g ...
    python latte.py --method ECoT --data_name credit-g ...
    python latte.py --method Evo --data_name credit-g ...
    python latte.py --method ToT --data_name credit-g ...
"""

import utils
import Preprocess
import Serialization
import Postprocessor
import Retriever
import Selector
import Evaluator
import json
import os
import logging
import time
import argparse
import History_DB as hdb
import cRPN
import re
import numpy as np
import pandas
import ltm

VALID_METHODS = ['CoT', 'LtM', 'Critic', 'OPROc', 'OPRO', 'ECoT', 'Evo', 'ToT']


# ============================================================
# Argument Parsing (superset of all methods)
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description='Unified LLM4FE Framework')

    # Method selection
    parser.add_argument('--method', type=str, default='CoT', choices=VALID_METHODS,
                        help='Method to run')

    # Log
    parser.add_argument('--log_path', type=str, default='./log')
    parser.add_argument('--log_filename', type=str, default=None)

    # Main
    parser.add_argument('--data_name', type=str, default='credit-g')
    parser.add_argument('--output_format', type=str, default='NL',
                        choices=['NL', 'Rule', 'Code', 'cRPN'])
    parser.add_argument('--llm_model', type=str, default='deepseek-chat')
    parser.add_argument('--nl2code', type=str, default='')
    parser.add_argument('--code_model', type=str, default='deepseek-chat')
    parser.add_argument('--enlarge_num', type=int, default=3)
    parser.add_argument('--selector', type=int, default=1)
    parser.add_argument('--history', type=int, default=1)
    parser.add_argument('--top', type=int, default=0,
                        help='Use Top-k history feedback for CoT/LtM; 0 disables, >0 sets k')
    parser.add_argument('--iter', type=int, default=10)

    # Preprocess
    parser.add_argument('--task_type', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

    # Serialization
    parser.add_argument('--ensemble', type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=4)
    parser.add_argument('--sample_method', type=int, default=1)
    parser.add_argument('--demo_format', type=int, default=0)
    parser.add_argument('--op_type', type=int, default=2)
    parser.add_argument('--metadata_cat', type=int, default=3)

    # LLM query
    parser.add_argument('--max_tokens', type=int, default=8192)
    parser.add_argument('--temperature', type=float, default=1.0)

    # OPRO / OPROc: dialogue turns
    parser.add_argument('--dialogue_turns', type=int, default=10)

    # ECoT / Evo: EA params
    parser.add_argument('--ind_num', type=int, default=90)
    parser.add_argument('--remove_time', type=int, default=1)
    parser.add_argument('--update_time', type=int, default=2)
    parser.add_argument('--random_sample', type=int, default=0,
                        help='Whether to use random sampling in collect_data (1=yes, 0=no)')

    # ToT params
    parser.add_argument('--num_thoughts', type=int, default=2)
    parser.add_argument('--max_steps', type=int, default=5)
    parser.add_argument('--max_states', type=int, default=1)
    parser.add_argument('--pruning_threshold', type=float, default=0.003)
    parser.add_argument('--search_strategy', type=str, default='mcts',
                        choices=['mcts', 'greedy'],
                        help='ToT node selection strategy')
    parser.add_argument('--model_type', type=str, default='auto',
                        choices=['decision_tree', 'random_forest', 'knn', 'mlp', 'auto'])
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default=None)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--n_neighbors', type=int, default=5)
    parser.add_argument('--hidden_layer_sizes', type=str, default='100')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_iter', type=int, default=200)

    return parser.parse_args()


# ============================================================
# Common Helpers
# ============================================================

def setup_logging(log_path, log_filename):
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
    return logger


def append_instruct(prompt, output_format):
    """Append output-format-specific instruction template to prompt."""
    mapping = {
        'NL': 'templates/instruct_NL.txt',
        'Rule': 'templates/instruct_rule.txt',
        'Code': 'templates/instruct_code.txt',
        'cRPN': 'templates/instruct_cRPN.txt',
    }
    if output_format in mapping:
        prompt += utils.read_txt(mapping[output_format])
    return prompt


def exec_output(output_format, llm_output, data_name, data, target=None):
    """Execute LLM output on a single dataframe, return (success_ops, new_data)."""
    if output_format == 'NL':
        return Postprocessor.NL2exec(llm_output, data_name, data)
    elif output_format == 'cRPN':
        return Postprocessor.RPN2exec(llm_output, data_name, data)
    elif output_format == 'Code':
        return Postprocessor.Code2exec(llm_output, data_name, data, target)
    return [], data


def exec_all_splits(output_format, llm_output, data_name, train_data, val_data, test_data, target=None):
    """Execute LLM output on train/val/test, return (success_ops, new_train, new_val, new_test)."""
    success_ops, new_train = exec_output(output_format, llm_output, data_name, train_data, target)
    _, new_val = exec_output(output_format, llm_output, data_name, val_data, target)
    _, new_test = exec_output(output_format, llm_output, data_name, test_data, target)
    return success_ops, new_train, new_val, new_test


def extract_ops_string(output_format, llm_output, success_ops):
    """Extract ops_string for history recording."""
    if output_format == 'NL':
        return Postprocessor.reconstruct_string(success_ops)
    elif output_format == 'cRPN':
        ops_string = llm_output.replace('\n', '')
        pattern = r'\{(.*?)\}'
        ops_string = re.findall(pattern, ops_string)
        if ops_string:
            ops_string = [s.replace('\\', '') for s in ops_string]
        return ops_string[0] if ops_string else ""
    elif output_format == 'Code':
        code_blocks = Postprocessor.extract_code_blocks(llm_output)
        return f"{code_blocks}"
    return ""


def feature_selection_rf(new_train, new_val, new_test, target, task_type, k, logger,
                         new_predictor=None, new_val_acc=None):
    """Run RF-based feature selection and reuse the current RF result when available."""
    if new_predictor is None or new_val_acc is None:
        new_predictor, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
        logger.info(f"new_val_acc = {new_val_acc}")

    val_importance = Selector.calculate_rf_importance(new_train, new_val, new_predictor, target, task_type)
    sel_train, dropped_columns = Selector.select_top_features(new_train, val_importance, k)
    sel_val, _ = Selector.select_top_features(new_val, val_importance, k)
    sel_test, _ = Selector.select_top_features(new_test, val_importance, k)
    logger.info(f"dropped columns = {dropped_columns}")

    if len(dropped_columns) > 0:
        sel_predictor, sel_val_acc = Evaluator.train_and_evaluate_rf(sel_train, sel_val, target, task_type)
        _, sel_test_acc = Evaluator.train_and_evaluate_rf(sel_train, sel_test, target, task_type)
        logger.info(f"sel_val_acc = {sel_val_acc}")
        logger.info(f"sel_test_acc = {sel_test_acc}")

        if sel_val_acc > new_val_acc:
            return sel_train, sel_val, sel_test, sel_val_acc, dropped_columns
    return new_train, new_val, new_test, new_val_acc, []


def save_improved(data_name, new_train, new_val, new_test, metadata, new_val_acc,
                  score_list, best_performance, best_metadata, logger):
    """Save data when improved. Returns updated (score_list, best_performance, best_metadata)."""
    if new_val_acc > score_list[-1]:
        score_list.append(new_val_acc)
        output_path = f'tmp/{data_name}/metadata.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info("--- metadata updated! ---")
        if new_val_acc > best_performance:
            best_performance = new_val_acc
            best_metadata = metadata
            new_train.to_csv(os.path.join(f"./tmp/{data_name}", "best_train.csv"), index=False)
            new_test.to_csv(os.path.join(f"./tmp/{data_name}", "best_test.csv"), index=False)
        new_train.to_csv(os.path.join(f"./tmp/{data_name}", "train.csv"), index=False)
        new_val.to_csv(os.path.join(f"./tmp/{data_name}", "val.csv"), index=False)
        new_test.to_csv(os.path.join(f"./tmp/{data_name}", "test.csv"), index=False)
    return score_list, best_performance, best_metadata


def final_summary(args, logger, target, task_type, score_list, best_performance,
                  best_metadata, total_token_usage, total_start_time,
                  initial_test_acc=None):
    """Log final summary."""
    total_end_time = time.time()
    try:
        train_data, test_data = Evaluator.best_dataset(args.data_name)
        predictor, test_acc = Evaluator.train_and_evaluate(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_ag = {test_acc}")
        predictor, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_rf = {test_acc}")
    except Exception as e:
        logger.warning(f"No best dataset found: {e}")
        logger.info(f"final_test_acc_ag = None")
        logger.info(f"final_test_acc_rf = None")
    logger.info(f"Total token usage = {total_token_usage}")
    logger.info(f"Score list = {score_list}")
    logger.info(f"Best performance = {best_performance}")
    logger.info(f"Best feature set = {best_metadata}")
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")


def make_serializer(args, target):
    return Serialization.Serializer(
        data_name=args.data_name,
        ensemble=args.ensemble,
        sample_size=args.sample_size,
        sample_method=args.sample_method,
        demo_format=args.demo_format,
        op_type=args.op_type,
        task_type=args.task_type,
        target=target,
        metadata_cat=args.metadata_cat
    )


# ============================================================
# Method: CoT
# ============================================================

def run_CoT(args, logger, df_train, target, task_type, k):
    store = hdb.ScoreStore()
    if args.top > 0:
        store.clear_table_data("history.db", "plans")
    total_start_time = time.time()

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    _, val_acc = Evaluator.train_and_evaluate_rf(train_data, val_data, target, task_type)
    _, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
    logger.info(f"val_acc = {val_acc}")
    logger.info(f"test_acc = {test_acc}")

    score_list = [val_acc]
    best_performance = val_acc
    best_metadata = None
    total_token_usage = 0
    history = []

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")

        ser = make_serializer(args, target)
        prompt_list = ser.generate_initial_prompt()
        prompt = prompt_list[0]

        # history feedback (only when not using top-k)
        if args.history and args.top <= 0:
            rejected_history = sorted([d for d in history if next(iter(d.values())) <= 0.0],
                                      key=lambda d: next(iter(d.values())), reverse=True)
            accepted_history = sorted([d for d in history if next(iter(d.values())) > 0.0],
                                      key=lambda d: next(iter(d.values())), reverse=True)
            prompt += f"\nAccepted features so far: \n{accepted_history}"
            prompt += f"\nRejected features so far: \n{rejected_history}"

        # top-k feedback (replaces history feedback when enabled)
        if args.top > 0 and i > 0:
            ops, rpns, scores = store.top_k(min(i, args.top))
            prompt += f"\nThese are the current top new features and their score (accuracy gain):\n"
            for j in range(len(rpns)):
                if scores[j] > 0:
                    if args.output_format == 'cRPN':
                        rpn_string = '{' + ", ".join(str(item) for item in rpns[j]) + '}'
                    elif args.output_format == 'NL':
                        rpn_string = rpns[j]
                    elif args.output_format == 'Code':
                        rpn_string = rpns[j]
                    prompt += f"top {j+1}: new features = {rpn_string}, score = {scores[j]}\n"

        logger.info(f"Initial prompt: {prompt}")
        prompt = append_instruct(prompt, args.output_format)

        llm_output, token_usage = utils.query_llm(prompt, max_tokens=args.max_tokens,
                                                    temperature=args.temperature, model=args.llm_model)
        total_token_usage += token_usage['total_tokens']
        logger.info(f"Token Usage:{token_usage}")
        llm_output = utils.remove_bold(llm_output)
        logger.info(f"LLM Output: {llm_output}")

        train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
        success_ops, new_train, new_val, new_test = exec_all_splits(
            args.output_format, llm_output, args.data_name, train_data, val_data, test_data, target)
        logger.info(f"Success Operators:\n{success_ops}")

        metadata = Postprocessor.exec_metadata(success_ops, args.data_name)
        logger.info(f"Extracted Metadata: {metadata}")

        new_predictor, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
        _, new_test_acc = Evaluator.train_and_evaluate_rf(new_train, new_test, target, task_type)
        logger.info(f"new_val_acc = {new_val_acc}")
        logger.info(f"new_test_acc = {new_test_acc}")

        if args.selector:
            new_train, new_val, new_test, new_val_acc, dropped = feature_selection_rf(
                new_train, new_val, new_test, target, task_type, k, logger,
                new_predictor=new_predictor, new_val_acc=new_val_acc)
            if dropped:
                metadata = Selector.update_metadata(metadata, dropped)

        ops_string = extract_ops_string(args.output_format, llm_output, success_ops)
        logger.info(f"ops_string = {ops_string}")
        history.append({ops_string: new_val_acc - score_list[-1]})

        # store top-k history
        if args.top > 0:
            if args.output_format == "cRPN":
                rpn = ops_string.split(',')
            else:
                rpn = [ops_string]
            store.add(metadata, success_ops, new_val_acc - score_list[-1], rpn)
            logger.info("---store history---")

        score_list, best_performance, best_metadata = save_improved(
            args.data_name, new_train, new_val, new_test, metadata, new_val_acc,
            score_list, best_performance, best_metadata, logger)

        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    store.close()
    final_summary(args, logger, target, task_type, score_list, best_performance,
                  best_metadata, total_token_usage, total_start_time, test_acc)


# ============================================================
# Method: LtM
# ============================================================

def run_LtM(args, logger, df_train, target, task_type, k):
    return ltm.run_LtM(
        args, logger, df_train, target, task_type, k,
        make_serializer=make_serializer,
        exec_all_splits=exec_all_splits,
        feature_selection_rf=feature_selection_rf,
        extract_ops_string=extract_ops_string,
        save_improved=save_improved,
        final_summary=final_summary,
    )


# ============================================================
# Method: Critic
# ============================================================

def run_Critic(args, logger, df_train, target, task_type, k):
    downstream = "rf"
    store = hdb.ScoreStore()
    total_start_time = time.time()

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    predictor, val_acc = Evaluator.train_and_evaluate_rf(train_data, val_data, target, task_type)
    _, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
    logger.info(f"val_acc = {val_acc}")
    logger.info(f"test_acc = {test_acc}")

    score_list = [val_acc]
    best_performance = val_acc
    best_metadata = None
    total_token_usage = 0
    history = []
    llm_output_critic = None

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")

        ser = make_serializer(args, target)
        prompt_list = ser.generate_initial_prompt()
        prompt = prompt_list[0]

        prompt_list = ser.generate_critic_prompt()
        prompt_critic = prompt_list[0]

        rejected_history = sorted([d for d in history if next(iter(d.values())) <= 0.0],
                                  key=lambda d: next(iter(d.values())), reverse=True)
        accepted_history = sorted([d for d in history if next(iter(d.values())) > 0.0],
                                  key=lambda d: next(iter(d.values())), reverse=True)

        logger.info(f"Initial prompt: {prompt}")
        logger.info(f"Initial critic prompt: {prompt_critic}")

        if i != 0 and llm_output_critic is not None:
            prompt += llm_output_critic

        prompt = append_instruct(prompt, args.output_format)

        llm_output, token_usage = utils.query_llm(prompt, max_tokens=args.max_tokens,
                                                    temperature=args.temperature, model=args.llm_model)
        total_token_usage += token_usage['total_tokens']
        logger.info(f"Token Usage:{token_usage}")
        llm_output = utils.remove_bold(llm_output)
        logger.info(f"LLM Output: {llm_output}")

        train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
        success_ops, new_train, new_val, new_test = exec_all_splits(
            args.output_format, llm_output, args.data_name, train_data, val_data, test_data, target)
        logger.info(f"Success Operators:\n{success_ops}")

        metadata = Postprocessor.exec_metadata(success_ops, args.data_name)
        logger.info(f"Extracted Metadata: {metadata}")

        new_predictor, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
        _, new_test_acc = Evaluator.train_and_evaluate_rf(new_train, new_test, target, task_type)
        logger.info(f"new_val_acc = {new_val_acc}")
        logger.info(f"new_test_acc = {new_test_acc}")

        if args.selector:
            new_train, new_val, new_test, new_val_acc, dropped = feature_selection_rf(
                new_train, new_val, new_test, target, task_type, k, logger,
                new_predictor=new_predictor, new_val_acc=new_val_acc)
            if dropped:
                metadata = Selector.update_metadata(metadata, dropped)

        ops_string = extract_ops_string(args.output_format, llm_output, success_ops)
        logger.info(f"ops_string = {ops_string}")
        history.append({ops_string: new_val_acc - score_list[-1]})

        # Critic feedback
        if new_val_acc > score_list[-1]:
            llm_output += f"\n Accept! Accuracy Gain = {new_val_acc - score_list[-1]}"
        else:
            llm_output += f"\n Rejected! Accuracy Gain = {new_val_acc - score_list[-1]}"

        llm_output_critic, critic_token = utils.query_critic(
            prompt, llm_output, prompt_critic,
            max_tokens=args.max_tokens, temperature=args.temperature, model=args.llm_model)
        total_token_usage += critic_token['total_tokens']
        logger.info(f"Token Usage:{critic_token}")
        llm_output_critic = utils.remove_bold(llm_output_critic)
        logger.info(f"Critic LLM Output: {llm_output_critic}")

        score_list, best_performance, best_metadata = save_improved(
            args.data_name, new_train, new_val, new_test, metadata, new_val_acc,
            score_list, best_performance, best_metadata, logger)

        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    store.close()
    final_summary(args, logger, target, task_type, score_list, best_performance,
                  best_metadata, total_token_usage, total_start_time, test_acc)


# ============================================================
# Method: OPROc
# ============================================================

def run_OPROc(args, logger, df_train, target, task_type, k):
    import utils_oct
    import pandas as pd

    downstream = "rf"
    total_start_time = time.time()

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    predictor, val_acc = Evaluator.train_and_evaluate_rf(train_data, val_data, target, task_type)
    _, initial_test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
    logger.info(f"Initial val_acc = {val_acc}")
    logger.info(f"Initial test_acc = {initial_test_acc}")

    score_list = [val_acc]
    best_performance = val_acc
    total_token_usage = 0

    xtrain, ytrain, xval, yval, xtest, ytest, feature_names, scalers = utils_oct.preprocess_dataframe(
        train_data, val_data, test_data, target)
    num_features = len(feature_names)
    logger.info(f"Number of features: {num_features}")

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")

        train_data_iter, val_data_iter, test_data_iter = Evaluator.load_dataset(args.data_name)
        xtrain, ytrain, xval, yval, xtest, ytest, feature_names_iter, _ = utils_oct.preprocess_dataframe(
            train_data_iter, val_data_iter, test_data_iter, target)

        # Update num_features to match current data (may change due to feature selection)
        num_features = len(feature_names_iter)
        logger.info(f"Current number of features: {num_features}")

        if task_type == 1:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(random_state=42)
        else:
            from sklearn.ensemble import RandomForestRegressor
            clf = RandomForestRegressor(random_state=42)
        clf.fit(xtrain, ytrain)

        importance = clf.feature_importances_
        sorting_imp = np.argsort(-importance)
        r0 = "x{:.0f} = [x{:.0f} * x{:.0f}]".format(num_features + 1, sorting_imp[0] + 1, sorting_imp[1] + 1)
        logger.info(f"Initial rule: {r0}")

        rule_text = utils_oct.rule_template(r0, num_features)
        exec(rule_text, globals())
        new_col = utils_oct.apply_rule(rule, xtrain, xval, xtest)
        new_xtrain, new_xval, new_xtest = utils_oct.add_column(xtrain, xval, xtest, new_col)
        _, train_acc, val_acc_init, test_acc_init = utils_oct.evaluate(
            new_xtrain, ytrain, new_xval, yval, new_xtest, ytest, task_type)
        best_CART = utils_oct.get_cart(new_xtrain, ytrain, new_xval, yval, args.seed, task_type)
        dt0 = utils_oct.tree_to_code(best_CART, ['x{}'.format(j) for j in range(1, len(new_xtrain[0]) + 1)])
        logger.info(f"Initial feature Val Acc: {val_acc_init:.4f}, Test Acc: {test_acc_init:.4f}")

        r_list = [r0]
        dt_list = [dt0]
        oct_score_list = [val_acc_init]
        train_score_list = [train_acc]
        oct_test_score_list = [test_acc_init]

        # OCT dialogue
        best_val_acc = np.max(np.array(oct_score_list))
        best_test_acc = oct_test_score_list[np.argmax(np.array(oct_score_list))]
        best_xtrain, best_xval, best_xtest = xtrain.copy(), xval.copy(), xtest.copy()
        pattern = r"x{}\s*=\s*\[.*?\]".format(num_features + 1)

        for turn in range(args.dialogue_turns):
            logger.info(f"--- Dialogue Turn {turn+1}/{args.dialogue_turns} ---")
            prompt = utils_oct.gen_prompt(r_list, dt_list, oct_score_list, num_features + 1, task_type)
            response, token_usage = utils.query_llm(prompt, max_tokens=args.max_tokens,
                                                     temperature=args.temperature, model=args.llm_model)
            total_token_usage += token_usage['total_tokens']
            logger.info(f"Turn {turn+1} LLM Output: {response}")

            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    extracted_text = match.group()
                    rule_text = utils_oct.rule_template(extracted_text, num_features)
                    exec(rule_text, globals())
                    new_col = utils_oct.apply_rule(rule, xtrain, xval, xtest)
                    new_xtrain2, new_xval2, new_xtest2 = utils_oct.add_column(xtrain, xval, xtest, new_col)
                    clf2, train_acc2, val_acc2, test_acc2 = utils_oct.evaluate(
                        new_xtrain2, ytrain, new_xval2, yval, new_xtest2, ytest, task_type)
                    best_CART2 = utils_oct.get_cart(new_xtrain2, ytrain, new_xval2, yval, args.seed, task_type)
                    dt2 = utils_oct.tree_to_code(best_CART2, ['x{}'.format(j) for j in range(1, len(new_xtrain2[0]) + 1)])
                    logger.info(f"Turn {turn+1} Val Acc: {val_acc2:.4f}, Test Acc: {test_acc2:.4f}")

                    r_list.append(extracted_text)
                    oct_score_list.append(val_acc2)
                    oct_test_score_list.append(test_acc2)
                    dt_list.append(dt2)
                    train_score_list.append(train_acc2)

                    if val_acc2 > best_val_acc:
                        best_val_acc = val_acc2
                        best_test_acc = test_acc2
                        best_xtrain = new_xtrain2.copy()
                        best_xval = new_xval2.copy()
                        best_xtest = new_xtest2.copy()
                    elif val_acc2 == best_val_acc and train_acc2 < train_score_list[np.argmax(np.array(oct_score_list[:-1]))]:
                        best_test_acc = test_acc2
                        best_xtrain = new_xtrain2.copy()
                        best_xval = new_xval2.copy()
                        best_xtest = new_xtest2.copy()
                except Exception as e:
                    logger.error(f"Turn {turn+1} failed: {e}")
            else:
                logger.warning(f"Turn {turn+1}: No valid rule pattern found")

        # Convert back to DataFrame
        best_train_df = pd.DataFrame(best_xtrain, columns=[f"f{j}" for j in range(best_xtrain.shape[1])])
        best_train_df[target] = ytrain
        best_val_df = pd.DataFrame(best_xval, columns=[f"f{j}" for j in range(best_xval.shape[1])])
        best_val_df[target] = yval
        best_test_df = pd.DataFrame(best_xtest, columns=[f"f{j}" for j in range(best_xtest.shape[1])])
        best_test_df[target] = ytest

        if args.selector:
            new_predictor, _ = Evaluator.train_and_evaluate_rf(best_train_df, best_val_df, target, task_type)
            val_importance = Selector.calculate_rf_importance(best_train_df, best_val_df, new_predictor, target, task_type)
            sel_train, dropped = Selector.select_top_features(best_train_df, val_importance, k)
            sel_val, _ = Selector.select_top_features(best_val_df, val_importance, k)
            sel_test, _ = Selector.select_top_features(best_test_df, val_importance, k)
            if len(dropped) > 0:
                _, sel_val_acc = Evaluator.train_and_evaluate_rf(sel_train, sel_val, target, task_type)
                if sel_val_acc > best_val_acc:
                    best_val_acc = sel_val_acc
                    best_train_df, best_val_df, best_test_df = sel_train, sel_val, sel_test

        if best_val_acc > score_list[-1]:
            score_list.append(best_val_acc)
            if best_val_acc > best_performance:
                best_performance = best_val_acc
                best_train_df.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_train.csv"), index=False)
                best_test_df.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_test.csv"), index=False)
            best_train_df.to_csv(os.path.join(f"./tmp/{args.data_name}", "train.csv"), index=False)
            best_val_df.to_csv(os.path.join(f"./tmp/{args.data_name}", "val.csv"), index=False)
            best_test_df.to_csv(os.path.join(f"./tmp/{args.data_name}", "test.csv"), index=False)
        else:
            train_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "train.csv"), index=False)
            val_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "val.csv"), index=False)
            test_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "test.csv"), index=False)

        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    total_end_time = time.time()
    try:
        train_data, test_data = Evaluator.best_dataset(args.data_name)
        predictor, test_acc = Evaluator.train_and_evaluate(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_ag = {test_acc}")
        predictor, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_rf = {test_acc}")
    except Exception as e:
        logger.warning(f"No best dataset found: {e}")
        logger.info(f"final_test_acc_ag = None")
        logger.info(f"final_test_acc_rf = None")
    logger.info(f"Total token usage = {total_token_usage}")
    logger.info(f"Score list = {score_list}")
    logger.info(f"Best performance = {best_performance}")
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")


# ============================================================
# Method: OPRO
# ============================================================

def OPRO_dialogue_query(initial_prompt, args, logger, train_data, val_data, test_data, target, task_type):
    dialogue_history = []
    best_val_acc = -float('inf')
    best_test_acc = -float('inf')
    best_output = None
    best_metadata = None
    best_train, best_val, best_test = None, None, None
    best_success_ops = None
    total_token_usage = 0

    for turn in range(args.dialogue_turns):
        logger.info(f"--- Dialogue Turn {turn+1}/{args.dialogue_turns} ---")
        if turn == 0:
            current_prompt = initial_prompt
        else:
            feedback = f"\n\nPrevious attempt (Turn {turn}):\n"
            feedback += f"Generated features: {dialogue_history[-1]['output_summary']}\n"
            feedback += f"Validation score: {dialogue_history[-1]['val_acc']:.4f}\n"
            if best_val_acc > -float('inf'):
                feedback += f"Best score so far: {best_val_acc:.4f}\n"
                feedback += f"Best features so far: {dialogue_history[dialogue_history[-1]['best_idx']]['output_summary']}\n"
            feedback += "\nPlease refine your feature generation to improve the validation score. "
            feedback += "Try different combinations, operations, or approaches that are totally different from previous attempts.\n"
            current_prompt = initial_prompt + feedback

        llm_output, token_usage = utils.query_llm(current_prompt, max_tokens=args.max_tokens,
                                                    temperature=args.temperature, model=args.llm_model)
        total_token_usage += token_usage['total_tokens']
        llm_output = utils.remove_bold(llm_output)
        logger.info(f"Turn {turn+1} LLM Output: {llm_output}")

        try:
            success_ops, new_train, new_val, new_test = exec_all_splits(
                args.output_format, llm_output, args.data_name,
                train_data.copy(), val_data.copy(), test_data.copy(), target)
            logger.info(f"Turn {turn+1} Success Operators:\n{success_ops}")
            metadata = Postprocessor.exec_metadata(success_ops, args.data_name)

            _, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
            _, new_test_acc = Evaluator.train_and_evaluate_rf(new_train, new_test, target, task_type)
            logger.info(f"Turn {turn+1} Val Acc: {new_val_acc:.4f}, Test Acc: {new_test_acc:.4f}")

            ops_string = extract_ops_string(args.output_format, llm_output, success_ops)

            dialogue_history.append({
                'turn': turn + 1, 'output': llm_output, 'output_summary': ops_string,
                'val_acc': new_val_acc, 'test_acc': new_test_acc, 'success_ops': success_ops,
                'metadata': metadata, 'new_train': new_train, 'new_val': new_val, 'new_test': new_test,
                'best_idx': len(dialogue_history) if new_val_acc > best_val_acc else (dialogue_history[-1]['best_idx'] if dialogue_history else 0)
            })

            if new_val_acc > best_val_acc:
                logger.info(f"*** Turn {turn+1}: New best: {new_val_acc:.4f} ***")
                best_val_acc = new_val_acc
                best_test_acc = new_test_acc
                best_output = llm_output
                best_metadata = metadata
                best_train, best_val, best_test = new_train, new_val, new_test
                best_success_ops = success_ops

        except Exception as e:
            logger.error(f"Turn {turn+1} failed: {e}")
            dialogue_history.append({
                'turn': turn + 1, 'output': llm_output, 'output_summary': 'FAILED',
                'val_acc': -float('inf'), 'test_acc': -float('inf'), 'error': str(e),
                'best_idx': dialogue_history[-1]['best_idx'] if dialogue_history else 0
            })

    return best_output, best_metadata, best_train, best_val, best_test, best_val_acc, best_test_acc, total_token_usage, best_success_ops


def run_OPRO(args, logger, df_train, target, task_type, k):
    downstream = "rf"
    store = hdb.ScoreStore()
    total_start_time = time.time()

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    predictor, val_acc = Evaluator.train_and_evaluate_rf(train_data, val_data, target, task_type)
    _, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
    logger.info(f"Initial val_acc = {val_acc}")
    logger.info(f"Initial test_acc = {test_acc}")

    score_list = [val_acc]
    best_performance = val_acc
    best_metadata = None
    total_token_usage = 0
    history = []

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")

        ser = make_serializer(args, target)
        prompt_list = ser.generate_initial_prompt()
        prompt = prompt_list[0]
        prompt = append_instruct(prompt, args.output_format)

        train_data_iter, val_data_iter, test_data_iter = Evaluator.load_dataset(args.data_name)
        (best_output, best_metadata_iter, best_train, best_val, best_test,
         best_val_acc, best_test_acc_turn, dialogue_tokens, best_success_ops) = OPRO_dialogue_query(
            prompt, args, logger, train_data_iter, val_data_iter, test_data_iter, target, task_type)
        total_token_usage += dialogue_tokens

        if best_val_acc > -float('inf'):
            new_predictor, _ = Evaluator.train_and_evaluate_rf(best_train, best_val, target, task_type)
            current_val_acc = best_val_acc

            if args.selector:
                val_importance = Selector.calculate_rf_importance(best_train, best_val, new_predictor, target, task_type)
                sel_train, dropped = Selector.select_top_features(best_train, val_importance, k)
                sel_val, _ = Selector.select_top_features(best_val, val_importance, k)
                sel_test, _ = Selector.select_top_features(best_test, val_importance, k)
                if len(dropped) > 0:
                    _, sel_val_acc = Evaluator.train_and_evaluate_rf(sel_train, sel_val, target, task_type)
                    if sel_val_acc > current_val_acc:
                        current_val_acc = sel_val_acc
                        best_metadata_iter = Selector.update_metadata(best_metadata_iter, dropped)
                        best_train, best_val, best_test = sel_train, sel_val, sel_test

            ops_string = extract_ops_string(args.output_format, best_output, best_success_ops)
            history.append({ops_string: current_val_acc - score_list[-1]})

            if current_val_acc > score_list[-1]:
                score_list.append(current_val_acc)
                output_path = f'tmp/{args.data_name}/metadata.json'
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(best_metadata_iter, f, ensure_ascii=False, indent=2)
                if current_val_acc > best_performance:
                    best_performance = current_val_acc
                    best_metadata = best_metadata_iter
                    best_train.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_train.csv"), index=False)
                    best_test.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_test.csv"), index=False)
                best_train.to_csv(os.path.join(f"./tmp/{args.data_name}", "train.csv"), index=False)
                best_val.to_csv(os.path.join(f"./tmp/{args.data_name}", "val.csv"), index=False)
                best_test.to_csv(os.path.join(f"./tmp/{args.data_name}", "test.csv"), index=False)
            else:
                train_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "train.csv"), index=False)
                val_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "val.csv"), index=False)
                test_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "test.csv"), index=False)
        else:
            logger.warning(f"Iteration {i+1}: No valid features generated in dialogue")
            train_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "train.csv"), index=False)
            val_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "val.csv"), index=False)
            test_data_iter.to_csv(os.path.join(f"./tmp/{args.data_name}", "test.csv"), index=False)

        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    store.close()
    final_summary(args, logger, target, task_type, score_list, best_performance,
                  best_metadata, total_token_usage, total_start_time, test_acc)


# ============================================================
# Method: ECoT
# ============================================================

def run_ECoT(args, logger, df_train, target, task_type, k):
    from island import island, island_group, prompt
    from Operation import show_ops_r, converge, check_valid, op_post_seq, collect_data

    downstream = "rf"
    store = hdb.ScoreStore()
    store.clear_table_data("history.db", "plans")
    total_start_time = time.time()

    ser = make_serializer(args, target)
    init_prompt_list = ser.generate_initial_prompt()
    init_prompt = init_prompt_list[0]
    if args.metadata_cat <= 1:
        init_prompt = ''

    collect_data(args.data_name, args.seed, random_sample=bool(args.random_sample))
    prompts = []
    accs = []
    df = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_train.csv')
    y = df.iloc[:, -1]
    df = df.iloc[:, :-1]

    # Calculate initial test accuracy for fallback
    df_test_init = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_test.csv')
    df_train_init = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_train.csv')
    _, initial_test_acc = Evaluator.train_and_evaluate_rf(df_train_init, df_test_init, 'target', task_type)
    logger.info(f"Initial test_acc = {initial_test_acc}")

    with open('prompt.txt', 'r') as f:
        for line in f:
            prompts.append(line + ' \n')
    with open('acc.txt', 'r') as f:
        for line in f:
            accs.append(float(line))

    max_acc = -100000
    baseline_acc = max(accs)
    best_prompt = prompts[np.argmax(accs)]
    total_token_usage = 0
    num_ge = 0
    island_g = island_group()
    new_island = island()
    for idx in range(len(prompts)):
        new_island.add_prompt(prompt(prompts[idx], accs[idx]))
        if idx % args.ind_num == args.ind_num - 1:
            island_g.add_island(new_island)
            new_island = island()
            new_island.add_prompt(prompt(prompts[idx], accs[idx]))
    logger.info(f"Construct {len(island_g.islands)} islands")

    system = ('you can transfer features to get a new feature set which is represented by postfix expression. '
              'Here are features (f0,f1...,fn) and opearations (sqrt, square, sin, cos, tanh, stand_scaler, '
              'minmax_scaler, quan_trans, sigmoid, log, reciprocal, cube, +, -, *, /). Everytime I will give you '
              'two new feature set examples (the latter, the better), please give me one better according them. '
              'Keep the format the same as in the example. Do not use original feature names.\n')

    acc_list = []
    num_change = 0
    best_filtered_features = None

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")
        now_num = len(prompts)

        for isl in island_g.islands:
            prt = isl.get_prompts()
            logger.info(f'input = {init_prompt + system + prt}')
            output, token_usage = utils.query_llm(
                prompt=init_prompt + system + prt,
                max_tokens=args.max_tokens, temperature=args.temperature, model=args.llm_model)
            logger.info(f"llm output = {output}")
            total_token_usage += token_usage['total_tokens']
            output = output.replace("\n", "")
            new_trans = re.findall(r'\[(.*?)\]', output)
            logger.info(f"new trans = {new_trans}")

            if len(new_trans) == 0:
                logger.info('no transformation')
            else:
                for trans in new_trans:
                    new_text = '[' + trans + ']'
                    if isl.is_repeat(new_text):
                        logger.info('no new transformation')
                    else:
                        logger.info(trans)
                        trans = trans.replace('f', '')
                        trans = trans.split(',')
                        is_valid = True
                        new_data = pandas.DataFrame()
                        for tran in trans:
                            try:
                                ops = show_ops_r(converge(tran.split()))
                                if not check_valid(ops):
                                    logger.info('no valid transformation (check_valid ops)')
                                    is_valid = False
                                    break
                                else:
                                    new_data[tran] = op_post_seq(df, ops)
                            except Exception as e:
                                logger.info(f'no valid transformation: {e}')
                                is_valid = False
                                break
                        if is_valid:
                            try:
                                new_data['target'] = y
                                train_data_new = new_data.copy()

                                df_val = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_val.csv')
                                y_val = df_val.iloc[:, -1]
                                df_val = df_val.iloc[:, :-1]

                                val_data_new = pandas.DataFrame()
                                for tran in trans:
                                    ops = show_ops_r(converge(tran.split()))
                                    val_data_new[tran] = op_post_seq(df_val, ops)
                                val_data_new['target'] = y_val
                                test_data_new = val_data_new

                                if args.selector:
                                    model, _ = Evaluator.train_and_evaluate_rf(train_data_new, test_data_new, 'target', task_type)
                                    importance = Selector.calculate_rf_importance(train_data_new, test_data_new, model, 'target', task_type)
                                    k_sel = len([imp for imp in importance if imp >= 0])
                                    if k_sel > 0:
                                        filtered_data, dropped_features = Selector.select_top_features(new_data, importance, k_sel)
                                        if len(dropped_features) > 0:
                                            logger.info(f"Dropped {len(dropped_features)} features: {dropped_features}")
                                        filtered_val_data = val_data_new[list(filtered_data.columns)]
                                    else:
                                        filtered_data = new_data.copy()
                                        filtered_val_data = val_data_new.copy()
                                else:
                                    filtered_data = new_data.copy()
                                    filtered_val_data = val_data_new.copy()

                                _, acc = Evaluator.train_and_evaluate_rf(filtered_data, filtered_val_data, 'target', task_type)
                                if acc > max_acc:
                                    num_change += 1
                                    max_acc = acc
                                    best_prompt = new_text
                                    best_filtered_features = list(filtered_data.columns[:-1])
                                    logger.info('higher accuracy is found!')
                                    filtered_data.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_train.csv"), index=False)
                                    filtered_val = val_data_new[list(filtered_data.columns)]
                                    filtered_val.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_val.csv"), index=False)

                                    df_test = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_test.csv')
                                    y_test = df_test.iloc[:, -1]
                                    df_test = df_test.iloc[:, :-1]
                                    test_data_transformed = pandas.DataFrame()
                                    for tran in trans:
                                        ops = show_ops_r(converge(tran.split()))
                                        test_data_transformed[tran] = op_post_seq(df_test, ops)
                                    test_data_transformed['target'] = y_test
                                    filtered_test = test_data_transformed[list(filtered_data.columns)]
                                    filtered_test.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_test.csv"), index=False)

                                new_prt = prompt(new_text, acc)
                                prompts.append(new_text)
                                accs.append(acc)
                                acc_list.append(acc)
                                isl.add_prompt(new_prt)
                                num_ge += 1
                                logger.info(f"new transformation found, acc={acc}, max_acc={max_acc}, changes={num_change}, gen={num_ge}")
                            except Exception as e:
                                logger.info(f'no valid transformation: {e}')

        if i % args.remove_time == args.remove_time - 1 and i != 0:
            for isl in island_g.islands:
                isl.remove()
        if i % args.update_time == args.update_time - 1 and i != 0:
            island_g.island_update()

        logger.info(f"{len(prompts)-now_num} transformations found in this generation")
        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    logger.info(f"prompts number = {len(prompts)}")
    logger.info(f"acc list = {acc_list}")
    logger.info(f"best prompt = {best_prompt}, best accuracy = {max_acc}")
    logger.info(f"Best performance = {max_acc}")

    # Final evaluation
    try:
        from autogluon.tabular import TabularDataset
        best_trans = re.findall(r'\[(.*?)\]', best_prompt)
        df_final = pandas.read_csv(f'data/{args.data_name}/seed{args.seed}/{args.data_name}_test.csv')
        y_final = df_final.iloc[:, -1]
        df_final = df_final.iloc[:, :-1]
        for trans in best_trans:
            new_text = '[' + trans + ']'
            trans = trans.replace('f', '')
            trans = trans.split(',')
            best_test = pandas.DataFrame()
            for tran in trans:
                ops = show_ops_r(converge(tran.split()))
                best_test[tran] = op_post_seq(df_final, ops)
            best_test['target'] = y_final

        if best_filtered_features is not None:
            available_features = [f for f in best_filtered_features if f in best_test.columns]
            best_test = best_test[available_features + ['target']]

        best_train = TabularDataset(f"./tmp/{args.data_name}/best_train.csv")
        predictor, test_acc = Evaluator.train_and_evaluate(best_train, best_test, "target", task_type)
        logger.info(f"final_test_acc_ag = {test_acc}")
        predictor, test_acc = Evaluator.train_and_evaluate_rf(best_train, best_test, "target", task_type)
        logger.info(f"final_test_acc_rf = {test_acc}")
    except Exception as e:
        logger.warning(f"No best dataset found: {e}")
        logger.info(f"final_test_acc_ag = None")
        logger.info(f"final_test_acc_rf = None")
    logger.info(f"Total token usage = {total_token_usage}")
    total_end_time = time.time()
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")


# ============================================================
# Method: Evo
# ============================================================

def run_Evo(args, logger, df_train, target, task_type, k):
    from island import island, island_group, prompt
    from autogluon.tabular import TabularDataset

    downstream = "rf"
    store = hdb.ScoreStore()
    store.clear_table_data("history.db", "plans")
    total_start_time = time.time()

    ser = make_serializer(args, target)
    init_prompt_list = ser.generate_initial_prompt()
    init_prompt = init_prompt_list[0]

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    _, baseline_acc = Evaluator.train_and_evaluate_rf(train_data, val_data, target, task_type)
    _, initial_test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
    logger.info(f"baseline_acc = {baseline_acc}")
    logger.info(f"Initial test_acc = {initial_test_acc}")

    prompts = []
    accs = []
    max_acc = baseline_acc
    best_prompt = None
    total_token_usage = 0
    num_ge = 0
    island_g = island_group()
    islands_initialized = False

    if args.output_format == 'NL':
        system = 'You are a feature engineering expert. I will give you two feature sets in natural language descriptions (the latter is better). Please generate one better feature set based on them. Keep the format the same as in the examples. Focus on creating meaningful transformations.\n'
    elif args.output_format == 'cRPN':
        system = ('you can transfer features to get a new feature set which is represented by postfix expression. '
                  'Here are features (f0,f1...,fn) and opearations (sqrt, square, sin, cos, tanh, stand_scaler, '
                  'minmax_scaler, quan_trans, sigmoid, log, reciprocal, cube, +, -, *, /). Everytime I will give you '
                  'two new feature set examples (the latter, the better), please give me one better according them. '
                  'Keep the format the same as in the example. Do not use original feature names.\n')
    elif args.output_format == 'Code':
        system = ('You are a feature engineering expert. I will give you two code snippets for feature generation '
                  '(the latter is better). Please generate one better code snippet based on them. The code should '
                  'define feature transformations that can be applied to a dataframe. Keep the format the same as in the examples.\n')

    instruct_mapping = {'NL': 'templates/instruct_NL.txt', 'cRPN': 'templates/instruct_cRPN.txt', 'Code': 'templates/instruct_code.txt'}

    acc_list = []
    num_change = 0
    best_metadata = None
    history = []

    for i in range(args.iter):
        iter_start = time.time()
        logger.info(f"========== Iteration {i+1}/{args.iter} ==========")
        now_num = len(prompts)

        if not islands_initialized and len(prompts) >= 5 * args.ind_num:
            logger.info(f"Initializing islands with {len(prompts)} samples...")
            new_island = island()
            for idx in range(len(prompts)):
                new_island.add_prompt(prompt(prompts[idx], accs[idx]))
                if (idx + 1) % args.ind_num == 0:
                    island_g.add_island(new_island)
                    new_island = island()
            if len(new_island.texts) > 0:
                island_g.add_island(new_island)
            islands_initialized = True
            logger.info(f"Constructed {len(island_g.islands)} islands")

        num_queries = len(island_g.islands) if islands_initialized else 1

        for query_idx in range(num_queries):
            if islands_initialized:
                isl = island_g.islands[query_idx]
                prt = isl.get_prompts()
                full_prompt = init_prompt + system + prt
            else:
                rejected_history = sorted([d for d in history if next(iter(d.values())) <= 0.0],
                                          key=lambda d: next(iter(d.values())), reverse=True)
                accepted_history = sorted([d for d in history if next(iter(d.values())) > 0.0],
                                          key=lambda d: next(iter(d.values())), reverse=True)
                full_prompt = init_prompt
                if accepted_history:
                    full_prompt += f"\nAccepted features so far: \n{accepted_history[:5]}\n"
                if rejected_history:
                    full_prompt += f"\nRejected features so far: \n{rejected_history[:5]}\n"

            full_prompt += utils.read_txt(instruct_mapping[args.output_format])

            output, token_usage = utils.query_llm(prompt=full_prompt, max_tokens=args.max_tokens,
                                                   temperature=args.temperature, model=args.llm_model)
            total_token_usage += token_usage['total_tokens']
            output = utils.remove_bold(output)
            logger.info(f"llm output = {output}")

            try:
                train_data_r, val_data_r, test_data_r = Evaluator.load_dataset(args.data_name)
                success_ops, new_train, new_val, new_test = exec_all_splits(
                    args.output_format, output, args.data_name, train_data_r, test_data_r, val_data_r, target)
                logger.info(f"Success Operators:\n{success_ops}")

                if len(success_ops) > 0:
                    metadata = Postprocessor.exec_metadata(success_ops, args.data_name)
                    ops_string = extract_ops_string(args.output_format, output, success_ops)
                    new_text = ops_string

                    is_duplicate = (isl.is_repeat(new_text) if islands_initialized else new_text in prompts)
                    if not is_duplicate:
                        new_predictor, new_val_acc = Evaluator.train_and_evaluate_rf(new_train, new_val, target, task_type)
                        _, new_test_acc = Evaluator.train_and_evaluate_rf(new_train, new_test, target, task_type)
                        logger.info(f"new_val_acc = {new_val_acc}, new_test_acc = {new_test_acc}")

                        if args.selector:
                            new_train, new_val, new_test, new_val_acc, dropped = feature_selection_rf(
                                new_train, new_val, new_test, target, task_type, k, logger,
                                new_predictor=new_predictor, new_val_acc=new_val_acc)
                            if dropped:
                                metadata = Selector.update_metadata(metadata, dropped)

                        acc_delta = new_val_acc - baseline_acc
                        history.append({ops_string: acc_delta})
                        new_prt = prompt(new_text + ' \n', new_val_acc)
                        prompts.append(new_text)
                        accs.append(new_val_acc)
                        acc_list.append(new_val_acc)
                        if islands_initialized:
                            isl.add_prompt(new_prt)
                        num_ge += 1

                        if new_val_acc > max_acc:
                            num_change += 1
                            max_acc = new_val_acc
                            best_prompt = new_text
                            best_metadata = metadata
                            logger.info('higher accuracy is found!')
                            new_train.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_train.csv"), index=False)
                            new_val.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_val.csv"), index=False)
                            new_test.to_csv(os.path.join(f"./tmp/{args.data_name}", "best_test.csv"), index=False)
                            output_path = f'tmp/{args.data_name}/metadata.json'
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(metadata, f, ensure_ascii=False, indent=2)

                        logger.info(f"acc={new_val_acc}, max_acc={max_acc}, changes={num_change}, gen={num_ge}")
            except Exception as e:
                logger.info(f'no valid transformation: {e}')

        if islands_initialized and i % args.remove_time == args.remove_time - 1 and i != 0:
            for isl in island_g.islands:
                isl.remove()
        if islands_initialized and i % args.update_time == args.update_time - 1 and i != 0:
            island_g.island_update()

        logger.info(f"{len(prompts)-now_num} transformations found in this generation")
        iter_end = time.time()
        logger.info(f"Time used for iteration {i+1}: {iter_end - iter_start:.2f} seconds")
        logger.info(f"Total token usage = {total_token_usage}")

    logger.info(f"prompts number = {len(prompts)}")
    logger.info(f"acc list = {acc_list}")
    logger.info(f"best prompt = {best_prompt}, best accuracy = {max_acc}")
    logger.info(f"Best performance = {max_acc}")
    logger.info(f"best metadata = {best_metadata}")

    best_train_path = f"./tmp/{args.data_name}/best_train.csv"
    best_test_path = f"./tmp/{args.data_name}/best_test.csv"
    if os.path.exists(best_train_path) and os.path.exists(best_test_path):
        try:
            best_train = TabularDataset(best_train_path)
            best_test = TabularDataset(best_test_path)
            predictor, test_acc = Evaluator.train_and_evaluate(best_train, best_test, target, task_type)
            logger.info(f"final_test_acc_ag = {test_acc}")
            predictor, test_acc = Evaluator.train_and_evaluate_rf(best_train, best_test, target, task_type)
            logger.info(f"final_test_acc_rf = {test_acc}")
        except Exception as e:
            logger.warning(f"Error evaluating best dataset: {e}")
            logger.info(f"final_test_acc_ag = None")
            logger.info(f"final_test_acc_rf = None")
    else:
        logger.warning("No best dataset found")
        logger.info(f"final_test_acc_ag = None")
        logger.info(f"final_test_acc_rf = None")

    logger.info(f"Total token usage = {total_token_usage}")
    total_end_time = time.time()
    logger.info(f"Total time used = {total_end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")
    store.close()


# ============================================================
# Method: ToT
# ============================================================

def run_ToT(args, logger, df_train, target, task_type, k):
    from tree_of_thoughts import ToTAgent, MonteCarloSearch

    downstream = "rf"
    total_start_time = time.time()

    train_data, val_data, test_data = Evaluator.load_dataset(args.data_name)
    if downstream == "rf":
        predictor, val_metrics = Evaluator.train_and_evaluate_all_rf(train_data, val_data, target, task_type)
        _, test_metrix = Evaluator.train_and_evaluate_all_rf(train_data, test_data, target, task_type)
        initial_test_acc, _, _, _ = test_metrix
    val_accuracy, _, _, _ = val_metrics
    logger.info(f"val_acc = {val_accuracy}")
    logger.info(f"test_acc = {initial_test_acc}")

    ser = make_serializer(args, target)
    task_str, features_str, data_sample = ser.generate_prompt_components()

    tot_txt = utils.read_txt("templates/instruct_ToT.txt")
    initial_prompt = task_str + f"\n{tot_txt}"
    mid_prompt = (features_str, data_sample)

    operators_simple = utils.read_txt_to_list("./templates/operators_simple.txt")
    operators_complex = utils.read_txt_to_list("./templates/operators_complex.txt")
    operators = operators_simple + operators_complex

    model_params = {}
    if args.model_type == 'decision_tree':
        model_params = {'max_depth': args.max_depth, 'min_samples_split': args.min_samples_split,
                        'min_samples_leaf': args.min_samples_leaf, 'max_features': args.max_features}
    elif args.model_type == 'random_forest':
        model_params = {'n_estimators': args.n_estimators, 'max_depth': args.max_depth,
                        'min_samples_split': args.min_samples_split, 'min_samples_leaf': args.min_samples_leaf}
    elif args.model_type == 'knn':
        model_params = {'n_neighbors': args.n_neighbors}
    elif args.model_type == 'mlp':
        hidden_sizes = tuple(map(int, args.hidden_layer_sizes.split(',')))
        model_params = {'hidden_layer_sizes': hidden_sizes, 'batch_size': args.batch_size, 'max_iter': args.max_iter}

    model = ToTAgent(strategy="cot", max_f=k, evaluation_strategy="value",
                     enable_react=False, k=args.num_thoughts, model=args.model_type,
                     llm_model=args.llm_model, output_format=args.output_format)

    tree_of_thoughts = MonteCarloSearch(model)

    solution = tree_of_thoughts.solve_feature_generation(
        data_name=args.data_name, max_f=k, ensemble=args.ensemble,
        sample_size=args.sample_size, sample_method=args.sample_method,
        demo_format=args.demo_format, op_type=args.op_type,
        task_type=args.task_type, target=target, metadata_cat=args.metadata_cat,
        num_thoughts=args.num_thoughts, max_steps=args.max_steps,
        max_states=args.max_states, pruning_threshold=args.pruning_threshold,
        trainingset=train_data, validationset=val_data, testset=test_data,
        initial_metrics=val_metrics, operations=operators,
        feature_names=df_train.columns.to_list(), model_params=model_params,
        use_selector=args.selector, use_history=args.history,
        search_strategy=args.search_strategy)

    end_time = time.time()
    try:
        train_data, test_data = Evaluator.best_dataset(args.data_name)
        predictor, test_acc = Evaluator.train_and_evaluate(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_ag = {test_acc}")
        predictor, test_acc = Evaluator.train_and_evaluate_rf(train_data, test_data, target, task_type)
        logger.info(f"final_test_acc_rf = {test_acc}")
    except Exception as e:
        logger.warning(f"No best dataset found: {e}")
        logger.info(f"final_test_acc_ag = None")
        logger.info(f"final_test_acc_rf = None")

    logger.info(f"Total time used = {end_time - total_start_time:.2f} seconds")
    logger.info("========== END ==========")


# ============================================================
# Main
# ============================================================

METHOD_RUNNERS = {
    'CoT': run_CoT,
    'LtM': run_LtM,
    'Critic': run_Critic,
    'OPROc': run_OPROc,
    'OPRO': run_OPRO,
    'ECoT': run_ECoT,
    'Evo': run_Evo,
    'ToT': run_ToT,
}


def main():
    args = parse_arguments()
    task_type = args.task_type

    utils.clear_files(f"./tmp/{args.data_name}/")

    # Default log filename
    if args.log_filename is None:
        args.log_filename = f"{args.data_name}_{args.method}_{args.llm_model}_{args.metadata_cat}_{args.seed}.log"

    logger = setup_logging(args.log_path, args.log_filename)
    logger.info(f"========== START {args.method} ==========")
    logger.info(f"Arguments: {vars(args)}")

    utils.copy_and_rename_metadata(args.data_name)

    df, df_train, df_test, df_val, target, label_list = Preprocess.split_and_prepare_data(
        data_name=args.data_name,
        task_type=args.task_type,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size
    )

    k = int(df_train.shape[1] * args.enlarge_num)

    runner = METHOD_RUNNERS[args.method]
    runner(args, logger, df_train, target, task_type, k)


if __name__ == "__main__":
    main()

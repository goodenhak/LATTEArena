import re
import time

import Evaluator
import History_DB as hdb
import Postprocessor
import Selector
import utils


def parse_ltm_subproblems(raw_text):
    """Parse ordered LtM subproblems from LLM output."""
    enumerated_subproblems = []
    fallback_subproblems = []
    skip_headers = {
        "ordered subproblems:",
        "subproblems:",
        "decomposition:",
        "decomposed subproblems:",
        "here are the ordered subproblems:",
    }

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue

        lowered = line.lower()
        if lowered in skip_headers:
            continue

        if re.match(r'^\s*(?:[-*]|step\s*\d+[\).:-]?|\d+[\).:-])\s*', line, flags=re.IGNORECASE):
            cleaned = re.sub(r'^\s*(?:[-*]|step\s*\d+[\).:-]?|\d+[\).:-])\s*', '', line, flags=re.IGNORECASE)
            if cleaned:
                enumerated_subproblems.append(cleaned)
            continue

        # Keep a conservative fallback only for lines that look like imperative tasks.
        if re.match(r'^(identify|decide|analyze|determine|generate|select|find|propose|evaluate|use)\b', lowered):
            fallback_subproblems.append(line)

    deduped = []
    candidates = enumerated_subproblems or fallback_subproblems
    for item in candidates:
        if item not in deduped:
            deduped.append(item)

    if deduped:
        return deduped

    fallback = raw_text.strip()
    return [fallback] if fallback else []


def build_ltm_base_prompt(args, target, history, store, iteration_idx, make_serializer):
    """Build the shared feature-engineering prompt body for LtM."""
    ser = make_serializer(args, target)
    prompt = ser.generate_initial_prompt()[0]

    if args.history and args.top <= 0:
        rejected_history = sorted([d for d in history if next(iter(d.values())) <= 0.0],
                                  key=lambda d: next(iter(d.values())), reverse=True)
        accepted_history = sorted([d for d in history if next(iter(d.values())) > 0.0],
                                  key=lambda d: next(iter(d.values())), reverse=True)
        prompt += f"\nAccepted features so far: \n{accepted_history}"
        prompt += f"\nRejected features so far: \n{rejected_history}"

    if args.top > 0 and iteration_idx > 0:
        ops, rpns, scores = store.top_k(min(iteration_idx, args.top))
        prompt += f"\nThese are the current top new features and their score (accuracy gain):\n"
        for j in range(len(rpns)):
            if scores[j] > 0:
                if args.output_format == 'cRPN':
                    rpn_string = '{' + ", ".join(str(item) for item in rpns[j]) + '}'
                elif args.output_format == 'NL':
                    rpn_string = rpns[j]
                elif args.output_format == 'Code':
                    rpn_string = rpns[j]
                else:
                    rpn_string = rpns[j]
                prompt += f"top {j+1}: new features = {rpn_string}, score = {scores[j]}\n"

    return prompt


def query_ltm_solution(base_prompt, args, logger):
    """Run Least-to-Most prompting and return the final feature-generation output."""
    total_tokens = 0

    decompose_template = utils.read_txt("templates/instruct_LtM_decompose.txt")
    solve_template = utils.read_txt("templates/instruct_LtM_solve.txt")
    output_instruction = utils.read_txt({
        'NL': 'templates/instruct_NL.txt',
        'Rule': 'templates/instruct_rule.txt',
        'Code': 'templates/instruct_code.txt',
        'cRPN': 'templates/instruct_cRPN.txt',
    }[args.output_format])

    decompose_prompt = f"{base_prompt}\n{decompose_template}"
    logger.info(f"LtM Decomposition Prompt: {decompose_prompt}")
    decomposition_output, token_usage = utils.query_llm(
        decompose_prompt, max_tokens=args.max_tokens, temperature=args.temperature, model=args.llm_model)
    total_tokens += token_usage['total_tokens']
    decomposition_output = utils.remove_bold(decomposition_output)
    logger.info(f"LtM Decomposition Token Usage:{token_usage}")
    logger.info(f"LtM Decomposition Output: {decomposition_output}")

    subproblems = parse_ltm_subproblems(decomposition_output)
    if not subproblems:
        subproblems = ["Generate the final candidate features for this task."]
    logger.info(f"LtM Parsed Subproblems: {subproblems}")

    sub_answers = []
    final_output = ""

    for idx, subproblem in enumerate(subproblems):
        history_text = "None yet."
        if sub_answers:
            history_lines = []
            for j, (prev_subproblem, prev_answer) in enumerate(sub_answers, start=1):
                history_lines.append(f"Subproblem {j}: {prev_subproblem}\nAnswer {j}: {prev_answer}")
            history_text = "\n\n".join(history_lines)

        current_instruction = (
            "Provide the final answer strictly in the required output format and do not add any extra text.\n"
            f"{output_instruction}"
            if idx == len(subproblems) - 1
            else "Answer this subproblem concisely so it can help solve the remaining subproblems."
        )

        solve_prompt = utils.fill_template({
            "[BASE_PROMPT]": base_prompt,
            "[PREVIOUS_QA]": history_text,
            "[CURRENT_SUBPROBLEM]": subproblem,
            "[CURRENT_INSTRUCTION]": current_instruction,
        }, solve_template)

        logger.info(f"LtM Solve Prompt {idx+1}: {solve_prompt}")
        answer, token_usage = utils.query_llm(
            solve_prompt, max_tokens=args.max_tokens, temperature=args.temperature, model=args.llm_model)
        total_tokens += token_usage['total_tokens']
        answer = utils.remove_bold(answer)
        logger.info(f"LtM Solve Token Usage {idx+1}:{token_usage}")
        logger.info(f"LtM Solve Output {idx+1}: {answer}")

        sub_answers.append((subproblem, answer))
        final_output = answer

    return final_output, total_tokens, subproblems, sub_answers


def run_LtM(args, logger, df_train, target, task_type, k, *,
            make_serializer, exec_all_splits, feature_selection_rf,
            extract_ops_string, save_improved, final_summary):
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

        base_prompt = build_ltm_base_prompt(args, target, history, store, i, make_serializer)
        logger.info(f"LtM Base Prompt: {base_prompt}")

        llm_output, ltm_tokens, _, _ = query_ltm_solution(base_prompt, args, logger)
        total_token_usage += ltm_tokens
        logger.info(f"LtM Final Output: {llm_output}")

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

import os
import re
import argparse


def check_content_has_warning(content):
    """
    Check if content has a warning/error.
    Returns True if warning/error occurred, False otherwise.
    """
    # Check for 'Warning' (exact case) or 'warnings' (case-insensitive)
    if 'Warning' in content:
        return True
    if 'warnings' in content.lower():
        return True
    if '警告' in content:
        return True
    return False


def get_folder_type(filepath):
    """
    Determine the folder type based on parent folder name.
    Returns 'opro' for O-prefixed folders, 'tot' for TM-prefixed folders,
    or 'default' for others.
    """
    parent_folder = os.path.basename(os.path.dirname(filepath))
    if parent_folder.startswith('O'):
        return 'opro'
    elif parent_folder.startswith('TM'):
        return 'tot'
    else:
        return 'default'


def parse_tot_log(filepath):
    """
    Parse a ToT-style log file (TM-prefixed folders).
    No iteration markers. Each round is from 'LLM Output:' to 'Success Operators:'.
    A round has an error if Warning/警告 appears between LLM Output and Success Operators.
    """
    with open(filepath, "r") as f:
        content = f.read()

    llm_output_pattern = r'LLM Output:'
    success_ops_pattern = r'Success Operators:'

    llm_matches = list(re.finditer(llm_output_pattern, content))
    success_matches = list(re.finditer(success_ops_pattern, content))

    if not llm_matches:
        return {
            "total_rounds": 0,
            "rounds_without_errors": 0,
            "rounds_with_errors": 0,
        }

    # Pair each LLM Output with the next Success Operators
    rounds_with_errors = 0
    total_rounds = 0

    for llm_match in llm_matches:
        llm_pos = llm_match.start()
        # Find the next Success Operators after this LLM Output
        success_pos = None
        for s_match in success_matches:
            if s_match.start() > llm_pos:
                success_pos = s_match.start()
                break

        if success_pos is None:
            continue

        total_rounds += 1
        segment = content[llm_pos:success_pos]

        if check_content_has_warning(segment):
            rounds_with_errors += 1

    rounds_without_errors = total_rounds - rounds_with_errors

    return {
        "total_rounds": total_rounds,
        "rounds_without_errors": rounds_without_errors,
        "rounds_with_errors": rounds_with_errors,
    }


def parse_log_for_success_rate(filepath):
    """Parse a single log file and return round-based success rate metrics."""
    folder_type = get_folder_type(filepath)

    if folder_type == 'tot':
        return parse_tot_log(filepath)

    with open(filepath, "r") as f:
        content = f.read()

    # Find iteration markers to identify rounds
    iteration_pattern = r'=+ Iteration (\d+)/(\d+) =+'
    iteration_matches = list(re.finditer(iteration_pattern, content))

    if not iteration_matches:
        return {
            "total_rounds": 0,
            "rounds_without_errors": 0,
            "rounds_with_errors": 0,
        }

    num_iterations = len(iteration_matches)

    if folder_type == 'opro':
        # For OPRO-style logs: each iteration has multiple dialogue turns
        # An iteration is successful if ANY turn in it succeeds (has no error)
        turn_pattern = r'--- Dialogue Turn (\d+)/(\d+) ---'
        iterations_with_errors = 0

        for i, iter_match in enumerate(iteration_matches):
            # Get iteration content boundaries
            iter_start = iter_match.start()
            if i + 1 < len(iteration_matches):
                iter_end = iteration_matches[i + 1].start()
            else:
                iter_end = len(content)

            iter_content = content[iter_start:iter_end]

            # Find all dialogue turns within this iteration
            turn_matches = list(re.finditer(turn_pattern, iter_content))

            if not turn_matches:
                # No turns found, check the whole iteration for errors
                if check_content_has_warning(iter_content):
                    iterations_with_errors += 1
                continue

            # Check each turn - iteration succeeds if ANY turn has no error
            any_turn_succeeded = False
            for j, turn_match in enumerate(turn_matches):
                turn_start = turn_match.start()
                if j + 1 < len(turn_matches):
                    turn_end = turn_matches[j + 1].start()
                else:
                    turn_end = len(iter_content)

                turn_content = iter_content[turn_start:turn_end]

                if not check_content_has_warning(turn_content):
                    # This turn succeeded (no error)
                    any_turn_succeeded = True
                    break  # One success is enough for the iteration

            if not any_turn_succeeded:
                iterations_with_errors += 1

        iterations_without_errors = num_iterations - iterations_with_errors

        return {
            "total_rounds": num_iterations,
            "rounds_without_errors": iterations_without_errors,
            "rounds_with_errors": iterations_with_errors,
        }
    else:
        # For default logs: each iteration is a single round
        rounds_with_errors = 0

        for i, match in enumerate(iteration_matches):
            start_pos = match.start()
            if i + 1 < len(iteration_matches):
                end_pos = iteration_matches[i + 1].start()
            else:
                end_pos = len(content)

            round_content = content[start_pos:end_pos]

            if check_content_has_warning(round_content):
                rounds_with_errors += 1

        rounds_without_errors = num_iterations - rounds_with_errors

        return {
            "total_rounds": num_iterations,
            "rounds_without_errors": rounds_without_errors,
            "rounds_with_errors": rounds_with_errors,
        }


def find_log_files(directory, datasets, seeds):
    """Find log files matching the given datasets and seeds."""
    files = []
    for root, _, filenames in os.walk(directory):
        for fname in sorted(filenames):
            if not fname.endswith(".log"):
                continue
            matched_dataset = None
            for ds in datasets:
                if fname.startswith(ds + "_"):
                    matched_dataset = ds
                    break
            if matched_dataset is None:
                continue
            for seed in seeds:
                if fname.endswith(f"_{seed}.log"):
                    files.append(os.path.join(root, fname))
                    break
    return files


def main():
    parser = argparse.ArgumentParser(description="Calculate success rate from log files.")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset name prefixes to include (e.g., credit-g kc1)")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Random seeds to include (e.g., 1 2 3)")
    args = parser.parse_args()

    files = find_log_files(args.directory, args.datasets, args.seeds)
    if not files:
        print("No matching log files found.")
        return

    total_rounds_all = 0
    total_success_rounds_all = 0

    for fpath in files:
        r = parse_log_for_success_rate(fpath)
        relative_path = os.path.relpath(fpath, args.directory)
        folder_type = get_folder_type(fpath)

        print(f"--- {relative_path} ---")
        if folder_type == 'opro':
            print(f"  Mode: OPRO (iteration-level success: any turn OK => iteration OK)")
        elif folder_type == 'tot':
            print(f"  Mode: ToT (each LLM Output -> Success Operators segment is a round)")
        print(f"  Total rounds: {r['total_rounds']}")
        print(f"  Rounds without errors: {r['rounds_without_errors']}")
        print(f"  Rounds with errors: {r['rounds_with_errors']}")

        if r['total_rounds'] > 0:
            file_rate = r['rounds_without_errors'] / r['total_rounds']
            print(f"  Success rate: {file_rate:.4f} ({r['rounds_without_errors']}/{r['total_rounds']})")
        else:
            print(f"  Success rate: N/A (no rounds found)")
        print()

        total_rounds_all += r['total_rounds']
        total_success_rounds_all += r['rounds_without_errors']

    # Summary
    print("=" * 50)
    print(f"Files processed: {len(files)}")
    print(f"Total rounds: {total_rounds_all}")
    print(f"Total rounds without errors: {total_success_rounds_all}")

    if total_rounds_all > 0:
        overall_rate = total_success_rounds_all / total_rounds_all
        print(f"Overall success rate: {overall_rate:.4f} ({total_success_rounds_all}/{total_rounds_all})")
    else:
        print("Overall success rate: N/A (no rounds found)")


if __name__ == "__main__":
    main()

import re
import os
import argparse
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum

class LogPattern(Enum):
    PATTERN_1 = "pattern1"  # TMN/T* methods: Token first, then acc
    PATTERN_2 = "pattern2"  # CGN etc: acc first, then token
    PATTERN_3 = "pattern3"  # OGR/OGC: Turn X Val Acc format


def detect_pattern(log_filename: str) -> Optional[LogPattern]:
    """
    Auto-detect the log pattern from file content.
    """
    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            content = f.read(5000)

            # Pattern 3: contains "Turn X Val Acc:" format
            if re.search(r'Turn \d+ Val Acc:', content):
                return LogPattern.PATTERN_3

            # Pattern 1: "Total tokens consumed in this batch:" appears before new_val_acc
            token_match = re.search(r'Total tokens consumed in this batch:', content)
            acc_match = re.search(r'new_val_acc\s*=', content)

            if token_match and acc_match:
                return LogPattern.PATTERN_1

            # Pattern 2: "Total token usage =" format, acc appears before token
            if re.search(r'Total token usage\s*=', content) and acc_match:
                return LogPattern.PATTERN_2

            # If only "token consumed" present without new_val_acc, also Pattern 1
            if token_match:
                return LogPattern.PATTERN_1

    except Exception as e:
        print(f"Error detecting pattern: {e}")

    return None


def process_pattern_1(log_filename: str, target_length: int = 20) -> Tuple[List[float], List[float]]:
    """
    Process Pattern 1 logs (TMN/T* methods).
    Token appears before accuracy: Total tokens consumed in this batch: XXX
    """
    print(f"--- Processing with Pattern 1: {log_filename} ---")

    file_data: Dict[str, List[Dict[str, Any]]] = {}
    current_file: Optional[str] = None

    FILE_HEADER_PATTERN = re.compile(r"========\s*(.+?)\s*========")
    ACC_LINE_PATTERN = re.compile(r".*-\s*INFO\s*-\s*(new_val_acc|sel_val_acc)\s*=\s*(\d+\.?\d*)")
    TOKEN_LINE_PATTERN = re.compile(r".*-\s*INFO\s*-\s*Total tokens consumed in this batch:\s*(\d+)")

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines.")

            current_token: Optional[int] = None
            current_accs: List[float] = []

            for line in lines:
                line = line.strip()

                header_match = FILE_HEADER_PATTERN.match(line)
                if header_match:
                    if current_file and current_token is not None:
                        max_acc = max(current_accs) if current_accs else 0.0
                        file_data[current_file].append({
                            'max_acc': max_acc,
                            'token_usage': current_token
                        })

                    current_file = header_match.group(1).strip()
                    file_data[current_file] = []
                    current_token = None
                    current_accs = []
                    continue

                token_match = TOKEN_LINE_PATTERN.match(line)
                if token_match and current_file:
                    if current_token is not None:
                        max_acc = max(current_accs) if current_accs else 0.0
                        file_data[current_file].append({
                            'max_acc': max_acc,
                            'token_usage': current_token
                        })
                        current_accs = []

                    current_token = int(token_match.group(1))
                    continue

                acc_match = ACC_LINE_PATTERN.match(line)
                if acc_match and current_file and current_token is not None:
                    value = acc_match.group(2)
                    current_accs.append(float(value))

            if current_file and current_token is not None:
                max_acc = max(current_accs) if current_accs else 0.0
                file_data[current_file].append({
                    'max_acc': max_acc,
                    'token_usage': current_token
                })

    except Exception as e:
        print(f"Error processing Pattern 1: {e}")
        return [], []

    return _calculate_final_results(file_data, target_length)


def process_pattern_2(log_filename: str, target_length: int = 20) -> Tuple[List[float], List[float]]:
    """
    Process Pattern 2 logs (CGN and similar methods).
    Accuracy appears before token: new_val_acc, sel_val_acc, Total token usage
    """
    print(f"--- Processing with Pattern 2: {log_filename} ---")

    file_data: Dict[str, List[Dict[str, Any]]] = {}
    current_file: Optional[str] = None

    FILE_HEADER_PATTERN = re.compile(r"========\s*(.+?)\s*========")
    DATA_LINE_PATTERN = re.compile(r".*-\s*INFO\s*-\s*(new_val_acc|sel_val_acc|Total token usage)\s*=\s*(\d+\.?\d*)")

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            current_iteration: Dict[str, Any] = {'new_acc': None, 'sel_acc': None, 'token_usage': None}

            for line in f:
                line = line.strip()

                header_match = FILE_HEADER_PATTERN.match(line)
                if header_match:
                    if current_file and current_iteration['new_acc'] is not None:
                        file_data[current_file].append(current_iteration)

                    current_file = header_match.group(1).strip()
                    file_data[current_file] = []
                    current_iteration = {'new_acc': None, 'sel_acc': None, 'token_usage': None}
                    continue

                data_match = DATA_LINE_PATTERN.match(line)
                if data_match and current_file:
                    key = data_match.group(1)
                    value = data_match.group(2)

                    if key == "new_val_acc":
                        if current_iteration['new_acc'] is not None:
                            file_data[current_file].append(current_iteration)
                        current_iteration = {'new_acc': float(value), 'sel_acc': None, 'token_usage': None}

                    elif key == "sel_val_acc" and current_iteration['new_acc'] is not None:
                        current_iteration['sel_acc'] = float(value)

                    elif key == "Total token usage" and current_iteration['new_acc'] is not None:
                        current_iteration['token_usage'] = int(float(value))

            if current_file and current_iteration['new_acc'] is not None:
                file_data[current_file].append(current_iteration)

    except Exception as e:
        print(f"Error processing Pattern 2: {e}")
        return [], []

    # Convert to unified data format
    converted_data: Dict[str, List[Dict[str, Any]]] = {}
    for filename, iterations in file_data.items():
        converted_data[filename] = []
        for iteration in iterations[:target_length]:
            new_acc = iteration['new_acc']
            sel_acc = iteration['sel_acc']
            token_usage = iteration['token_usage']

            max_pair_acc = max(new_acc, sel_acc if sel_acc is not None else new_acc)
            converted_data[filename].append({
                'max_acc': max_pair_acc,
                'token_usage': token_usage if token_usage is not None else 0
            })

    return _calculate_final_results(converted_data, target_length)


def process_pattern_3(log_filename: str, target_length: int = 20) -> Tuple[List[float], List[float]]:
    """
    Process Pattern 3 logs (OGR/OGC methods).
    Turn X Val Acc: XXX, Test Acc: XXX format
    """
    print(f"--- Processing with Pattern 3: {log_filename} ---")

    file_data: Dict[str, List[Dict[str, Any]]] = {}
    current_file: Optional[str] = None

    FILE_HEADER_PATTERN = re.compile(r"========\s*(.+?)\s*========")
    TURN_ACC_PATTERN = re.compile(r".*-\s*INFO\s*-\s*Turn\s+(\d+)\s+Val Acc:\s*([\d.]+)")
    BEST_RESULT_PATTERN = re.compile(r".*-\s*INFO\s*-\s*Best dialogue result.*Val Acc:\s*([\d.]+)")
    AFTER_SELECTION_PATTERN = re.compile(r".*-\s*INFO\s*-\s*After selection.*Val Acc:\s*([\d.]+)")
    TOKEN_PATTERN = re.compile(r".*-\s*INFO\s*-\s*Total token usage\s*=\s*(\d+)")

    try:
        with open(log_filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"Read {len(lines)} lines.")

            current_turn_accs: List[float] = []
            current_best_acc: Optional[float] = None
            current_after_sel_acc: Optional[float] = None
            last_token: int = 0

            for line in lines:
                line = line.strip()

                header_match = FILE_HEADER_PATTERN.match(line)
                if header_match:
                    current_file = header_match.group(1).strip()
                    file_data[current_file] = []
                    current_turn_accs = []
                    current_best_acc = None
                    current_after_sel_acc = None
                    last_token = 0
                    continue

                if not current_file:
                    continue

                turn_match = TURN_ACC_PATTERN.match(line)
                if turn_match:
                    acc = float(turn_match.group(2))
                    current_turn_accs.append(acc)
                    continue

                best_match = BEST_RESULT_PATTERN.match(line)
                if best_match:
                    current_best_acc = float(best_match.group(1))
                    continue

                after_sel_match = AFTER_SELECTION_PATTERN.match(line)
                if after_sel_match:
                    current_after_sel_acc = float(after_sel_match.group(1))
                    continue

                # Total token usage marks the end of an iteration
                token_match = TOKEN_PATTERN.match(line)
                if token_match:
                    token_usage = int(token_match.group(1))

                    # Skip duplicate token lines
                    if token_usage == last_token:
                        continue

                    # Compute max accuracy for this iteration
                    max_acc = 0.0
                    if current_turn_accs:
                        max_acc = max(current_turn_accs)
                    if current_best_acc is not None:
                        max_acc = max(max_acc, current_best_acc)
                    if current_after_sel_acc is not None:
                        max_acc = max(max_acc, current_after_sel_acc)

                    if max_acc > 0:
                        file_data[current_file].append({
                            'max_acc': max_acc,
                            'token_usage': token_usage
                        })

                    last_token = token_usage
                    current_turn_accs = []
                    current_best_acc = None
                    current_after_sel_acc = None

    except Exception as e:
        print(f"Error processing Pattern 3: {e}")
        return [], []

    return _calculate_final_results(file_data, target_length)


def _calculate_final_results(
    file_data: Dict[str, List[Dict[str, Any]]],
    target_length: int
) -> Tuple[List[float], List[float]]:
    """
    Compute average monotonic-max accuracy and token usage across all files.
    """
    if not file_data or all(not data for data in file_data.values()):
        print("Warning: No valid data parsed from the log file.")
        return [], []

    all_monotonic_max_accs: List[List[float]] = []
    all_token_usages: List[List[int]] = []
    max_length = 0

    for filename, iterations in file_data.items():
        if not iterations:
            print(f"Warning: File {filename} has no valid iteration data.")
            continue

        print(f"  Processing: {filename} ({len(iterations)} iterations)")

        monotonic_accs: List[float] = []
        current_tokens: List[int] = []
        cumulative_max = 0.0

        for iteration in iterations[:target_length]:
            acc = iteration['max_acc']
            token = iteration['token_usage']

            cumulative_max = max(cumulative_max, acc)
            monotonic_accs.append(cumulative_max)
            current_tokens.append(token)

        if monotonic_accs:
            all_monotonic_max_accs.append(monotonic_accs)
            all_token_usages.append(current_tokens)
            max_length = max(max_length, len(monotonic_accs))

    if max_length == 0:
        return [], []

    # Pad shorter arrays to max_length using last value
    print(f"\n--- Padding to max iteration count: {max_length} ---")
    for i, (acc_array, token_array) in enumerate(zip(all_monotonic_max_accs, all_token_usages)):
        current_length = len(acc_array)
        if current_length < max_length:
            last_acc = acc_array[-1]
            last_token = token_array[-1]
            num_to_fill = max_length - current_length
            acc_array.extend([last_acc] * num_to_fill)
            token_array.extend([last_token] * num_to_fill)
            print(f"  File {i+1} ({current_length} pts) padded with {num_to_fill} pts")

    # Compute averages
    num_files = len(all_monotonic_max_accs)
    sum_accs = [0.0] * max_length
    sum_tokens = [0.0] * max_length

    for acc_array, token_array in zip(all_monotonic_max_accs, all_token_usages):
        for i in range(max_length):
            sum_accs[i] += acc_array[i]
            sum_tokens[i] += token_array[i]

    final_avg_acc_array = [100 * sum_accs[i] / num_files for i in range(max_length)]
    final_avg_token_array = [sum_tokens[i] / num_files for i in range(max_length)]

    print("\n========================================")
    print(f"Processed {num_files} files in total.")
    print("----------------------------------------")
    print(f"Average monotonic-max accuracy ({len(final_avg_acc_array)} pts):")
    print([round(x, 2) for x in final_avg_acc_array])
    print("----------------------------------------")
    print(f"Average token usage ({len(final_avg_token_array)} pts):")
    print([round(x, 1) for x in final_avg_token_array])
    print("========================================")

    return final_avg_acc_array, final_avg_token_array


def process_extracted_log(
    log_filename: str = "extracted_data.log",
    pattern: Optional[LogPattern] = None,
    target_length: int = 20
) -> Tuple[List[float], List[float]]:
    """
    Process an extracted log file, auto-detecting or using a specified pattern.

    Args:
        log_filename: Input log filename.
        pattern: Log pattern to use. If None, auto-detect.
        target_length: Maximum number of data points per file.

    Returns:
        Tuple of (avg_accuracy_array, avg_token_array).
    """
    if not os.path.exists(log_filename):
        print(f"Error: File '{log_filename}' not found.")
        return [], []

    # Auto-detect pattern if not specified
    if pattern is None:
        pattern = detect_pattern(log_filename)
        if pattern is None:
            print("Error: Could not auto-detect log pattern. Please specify --pattern manually.")
            return [], []
        print(f"Auto-detected pattern: {pattern.value}")

    if pattern == LogPattern.PATTERN_1:
        return process_pattern_1(log_filename, target_length)
    elif pattern == LogPattern.PATTERN_2:
        return process_pattern_2(log_filename, target_length)
    else:  # LogPattern.PATTERN_3
        return process_pattern_3(log_filename, target_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified log processor for three different log formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pattern descriptions:
  pattern1 - TMN/T* methods: Token appears before accuracy
             Format: Total tokens consumed in this batch: XXX
                     new_val_acc = XXX
                     sel_val_acc = XXX

  pattern2 - CGN and similar methods: Accuracy appears before token
             Format: new_val_acc = XXX
                     sel_val_acc = XXX
                     Total token usage = XXX

  pattern3 - OGR/OGC methods: Turn-based format
             Format: Turn X Val Acc: XXX, Test Acc: XXX
                     Best dialogue result - Val Acc: XXX
                     After selection - Val Acc: XXX
                     Total token usage = XXX

Examples:
  python extract_log_unified.py                          # auto-detect pattern
  python extract_log_unified.py -p pattern1              # use pattern1
  python extract_log_unified.py -i other.log -p pattern2 # specify input and pattern
        """
    )

    parser.add_argument(
        "-i", "--input",
        default="extracted_data.log",
        help="Input log filename (default: extracted_data.log)"
    )
    parser.add_argument(
        "-p", "--pattern",
        choices=["pattern1", "pattern2", "pattern3", "auto"],
        default="auto",
        help="Log pattern (default: auto)"
    )
    parser.add_argument(
        "-l", "--length",
        type=int,
        default=20,
        help="Max number of data points per file (default: 20)"
    )

    args = parser.parse_args()

    pattern = None
    if args.pattern != "auto":
        pattern = LogPattern(args.pattern)

    avg_accs, avg_tokens = process_extracted_log(
        log_filename=args.input,
        pattern=pattern,
        target_length=args.length
    )

import os
import re
import argparse
from collections import defaultdict

# ============================================================
# Manually specify initial AutoGluon accuracy for each (dataset, seed).
# Format: AG_INIT[(dataset_name, seed)] = value
# ============================================================
AG_INIT = {
    # Example entries — fill in your actual values:
    ("credit-g", 1): 0.77,
    ("credit-g", 2): 0.735,
    ("credit-g", 3): 0.8,
    ("credit-g", 4): 0.775,
    ("credit-g", 5): 0.75,
    ("credit-g", 6): 0.72,
    ("credit-approval", 1): 0.8696,
    ("credit-approval", 2): 0.8478,
    ("credit-approval", 3): 0.8623,
    ("credit-approval", 4): 0.8986,
    ("credit-approval", 5): 0.9058,
    ("credit-approval", 6): 0.8913,
    ("kc1", 1): 0.8720,
    ("kc1", 2): 0.8483,
    ("kc1", 3): 0.8673,
    ("kc1", 4): 0.8507,
    ("kc1", 5): 0.8578,
    ("kc1", 6): 0.8673,
    ("qsar-biodeg", 1): 0.8957,
    ("qsar-biodeg", 2): 0.8673,
    ("qsar-biodeg", 3): 0.8531,
    ("qsar-biodeg", 4): 0.8815,
    ("qsar-biodeg", 5): 0.8910,
    ("qsar-biodeg", 6): 0.9052,
    ("vehicle", 1): 0.7941,
    ("vehicle", 2): 0.7882,
    ("vehicle", 3): 0.7765,
    ("vehicle", 4): 0.7353,
    ("vehicle", 5): 0.7882,
    ("vehicle", 6): 0.7647,
    ("heart-h", 1): 0.8475,
    ("heart-h", 2): 0.8136,
    ("heart-h", 3): 0.7458,
    ("heart-h", 4): 0.8305,
    ("heart-h", 5): 0.7797,
    ("heart-h", 6): 0.7288,
    ("socmob", 1): 0.9483,
    ("socmob", 2): 0.9569,
    ("socmob", 3): 0.9397,
    ("socmob", 4): 0.9569,
    ("socmob", 5): 0.9440,
    ("socmob", 6): 0.9569,
    ("electricity", 1): 0.9315,
    ("electricity", 2): 0.9346,
    ("nomao", 1): 0.9727,
    ("nomao", 2): 0.9742,
}


def parse_log(filepath):
    """Parse a single log file and return extracted metrics."""
    with open(filepath, "r") as f:
        content = f.read()

    result = {}

    # Check for error pattern at end of file
    if re.search(r"ERROR: No columns to parse from file\s*\n=+ END =+\s*$", content):
        result["parse_error"] = True
        return result

    # Initial val_acc and test_acc (first occurrences after START)
    m = re.search(r"INFO - val_acc = ([\-\d.]+)", content)
    if m:
        result["init_val_acc"] = float(m.group(1))
    m = re.search(r"INFO - test_acc = ([\-\d.]+)", content)
    if m:
        result["init_test_acc"] = float(m.group(1))

    # Best performance (final validation accuracy)
    # Format 1: "INFO - Best performance = X.XX"
    # Format 2 (TMN): "INFO -     Accuracy Test: X.XXXX"
    m = re.search(r"INFO - Best performance = ([\-\d.]+)", content)
    if m:
        result["best_val_acc"] = float(m.group(1))
    else:
        m = re.search(r"INFO -\s+Accuracy Test: ([\-\d.]+)", content)
        if m:
            result["best_val_acc"] = float(m.group(1))

    # final_test_acc — from INFO log line or TMN format
    # Format 1: "INFO - final_test_acc = X.XX"
    # Format 2 (TMN): "rf final_test_acc = X.XX"
    m = re.search(r"INFO - final_test_acc = ([\-\d.]+)", content)
    if m:
        result["final_test_acc"] = float(m.group(1))
    else:
        m = re.search(r"^rf final_test_acc = ([\-\d.]+)", content, re.MULTILINE)
        if m:
            result["final_test_acc"] = float(m.group(1))

    # Total token usage (last occurrence)
    # Format 1: "INFO - Total token usage = XXXX"
    # Format 2 (TMN): "INFO - Total tokens consumed in this batch: XXXX"
    matches = re.findall(r"INFO - Total token usage = ([\d.]+)", content)
    if matches:
        result["total_tokens"] = float(matches[-1])
    else:
        matches = re.findall(r"INFO - Total tokens consumed in this batch: ([\d.]+)", content)
        if matches:
            result["total_tokens"] = float(matches[-1])

    # Total time used
    m = re.search(r"INFO - Total time used = ([\d.]+) seconds", content)
    if m:
        result["total_time"] = float(m.group(1))

    # final_test_acc_ag — plain text line outside log format
    # Format 1: "final_test_acc_ag = X.XX"
    # Format 2 (TMN): "ag final_test_acc = X.XX"
    m = re.search(r"^final_test_acc_ag = ([\-\d.]+)", content, re.MULTILINE)
    if m:
        result["final_test_acc_ag"] = float(m.group(1))
    else:
        m = re.search(r"^ag final_test_acc = ([\-\d.]+)", content, re.MULTILINE)
        if m:
            result["final_test_acc_ag"] = float(m.group(1))

    # Extract dataset name and seed from Arguments line
    m = re.search(r"'data_name': '([^']+)'", content)
    if m:
        result["data_name"] = m.group(1)
    m = re.search(r"'seed': (\d+)", content)
    if m:
        result["seed"] = int(m.group(1))

    return result


def find_log_files(directory, datasets, seeds):
    """Find log files matching the given datasets and seeds."""
    files = []
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(".log"):
            continue
        # Check dataset prefix
        matched_dataset = None
        for ds in datasets:
            if fname.startswith(ds + "_"):
                matched_dataset = ds
                break
        if matched_dataset is None:
            continue
        # Check seed suffix: filename ends with _{seed}.log
        for seed in seeds:
            if fname.endswith(f"_{seed}.log"):
                files.append(os.path.join(directory, fname))
                break
    return files


def main():
    parser = argparse.ArgumentParser(description="Extract and summarize log file metrics.")
    parser.add_argument("directory", help="Directory containing log files")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name prefix to include (e.g., credit-g)")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Random seeds to include (e.g., 1 2 3)")
    args = parser.parse_args()

    files = find_log_files(args.directory, [args.dataset], args.seeds)
    if not files:
        print("No matching log files found.")
        return

    # Collect metrics
    val_changes = []
    test_changes = []
    tokens = []
    times = []
    ag_changes = []
    # Track missing data
    val_missing = []
    test_missing = []
    ag_missing = []

    for fpath in files:
        r = parse_log(fpath)
        fname = os.path.basename(fpath)

        # Handle parse error case: set all acc improvements to 0
        if r.get("parse_error"):
            print(f"--- {fname} ---")
            print(f"  ERROR: No columns to parse from file - setting all acc changes to 0")
            val_changes.append(0)
            test_changes.append(0)
            ag_changes.append(0)
            print()
            continue

        init_val = r.get("init_val_acc")
        best_val = r.get("best_val_acc")
        init_test = r.get("init_test_acc")
        final_test = r.get("final_test_acc")
        final_ag = r.get("final_test_acc_ag")
        data_name = r.get("data_name")
        seed = r.get("seed")

        print(f"--- {fname} ---")
        if init_val is not None and best_val is not None:
            delta_val = best_val - init_val
            val_changes.append(delta_val)
            print(f"  Val acc: {init_val:.6f} -> {best_val:.6f}  (Δ={delta_val:+.6f})")
        else:
            val_missing.append(fname)
            print(f"  Val acc: MISSING DATA")

        if init_test is not None and final_test is not None:
            delta_test = final_test - init_test
            test_changes.append(delta_test)
            print(f"  Test acc: {init_test:.6f} -> {final_test:.6f}  (Δ={delta_test:+.6f})")
        else:
            test_missing.append(fname)
            print(f"  Test acc: MISSING DATA")

        if r.get("total_tokens") is not None:
            tokens.append(r["total_tokens"])
            print(f"  Tokens: {r['total_tokens']:.0f}")

        if r.get("total_time") is not None:
            times.append(r["total_time"])
            print(f"  Time: {r['total_time']:.2f}s")

        if final_ag is not None:
            ag_key = (data_name, seed) if data_name and seed is not None else None
            ag_init = AG_INIT.get(ag_key) if ag_key else None
            if ag_init is not None:
                delta_ag = final_ag - ag_init
                ag_changes.append(delta_ag)
                print(f"  AG acc: {ag_init:.6f} -> {final_ag:.6f}  (Δ={delta_ag:+.6f})")
            else:
                print(f"  AG acc: {final_ag:.6f}  (no initial AG value specified)")
        else:
            # Check if only final_test_acc_ag is missing (other fields are complete)
            other_fields_complete = all([
                init_val is not None,
                best_val is not None,
                init_test is not None,
                final_test is not None,
                r.get("total_tokens") is not None,
                r.get("total_time") is not None,
            ])
            if other_fields_complete:
                ag_changes.append(0)
                print(f"  AG acc: MISSING (only final_test_acc_ag missing, treating as Δ=0)")
            else:
                ag_missing.append(fname)
                print(f"  AG acc: NOT AVAILABLE")
        print()

    # Summary
    print("=" * 50)
    print(f"Files processed: {len(files)}")
    if val_changes:
        if len(val_changes) > 1:
            val_changes_trimmed = sorted(val_changes)[1:]  # Remove lowest
            print(f"Mean val acc change:  {sum(val_changes_trimmed)/len(val_changes_trimmed):+.6f}")
        else:
            print(f"Mean val acc change:  {sum(val_changes)/len(val_changes):+.6f}  (n={len(val_changes)})")
        print(f"Max val acc change:   {max(val_changes):+.6f}")
    if val_missing:
        print(f"  !! Val acc missing in {len(val_missing)} file(s): {', '.join(val_missing)}")
    if test_changes:
        if len(test_changes) > 1:
            test_changes_trimmed = sorted(test_changes)[1:]  # Remove lowest
            print(f"Mean test acc change: {sum(test_changes_trimmed)/len(test_changes_trimmed):+.6f}")
        else:
            print(f"Mean test acc change: {sum(test_changes)/len(test_changes):+.6f}  (n={len(test_changes)})")
        print(f"Max test acc change:  {max(test_changes):+.6f}")
    if test_missing:
        print(f"  !! Test acc missing in {len(test_missing)} file(s): {', '.join(test_missing)}")
    if tokens:
        print(f"Avg token cost:       {sum(tokens)/len(tokens):.0f}  (n={len(tokens)})")
    if times:
        print(f"Avg time used:        {sum(times)/len(times):.2f}s  (n={len(times)})")
    if ag_changes:
        if len(ag_changes) > 1:
            ag_changes_trimmed = sorted(ag_changes)[1:]  # Remove lowest
            print(f"Mean AG acc change:   {sum(ag_changes_trimmed)/len(ag_changes_trimmed):+.6f}")
        else:
            print(f"Mean AG acc change:   {sum(ag_changes)/len(ag_changes):+.6f}  (n={len(ag_changes)})")
        print(f"Max AG acc change:    {max(ag_changes):+.6f}")
    elif any(parse_log(f).get("final_test_acc_ag") is not None for f in files):
        print("Mean AG acc change:   N/A (initial AG values not specified in AG_INIT)")
    if ag_missing:
        print(f"  !! AG acc not available in {len(ag_missing)} file(s): {', '.join(ag_missing)}")


if __name__ == "__main__":
    main()

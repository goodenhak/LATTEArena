import os
import re
import math
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
    # Regression datasets (metric = 1 - RMSE, higher is better)
    ("bike_sharing", 1): -36.36,
    ("bike_sharing", 2): -38.26,
    ("bike_sharing", 3): -36.71,
    ("cpu_small", 1): -1.55,
    ("cpu_small", 2): -1.51,
    ("cpu_small", 3): -1.55,
    ("diamonds", 1): -514.22,
    ("diamonds", 2): -518.57,
    ("diamonds", 3): -528.23,
    ("wine_quality", 1): 0.39,
    ("wine_quality", 2): 0.39,
    ("wine_quality", 3): 0.40,
}


def _normalize_acc(value):
    """Normalize an accuracy-like value to fraction scale.

    Some logs print percents (77.0) instead of fractions (0.77). Accuracy
    fractions are always <= 1, and the regression metric 1 - RMSE is also
    always <= 1, so any value > 1 must be a percent.
    """
    if value > 1.0:
        return value / 100.0
    return value


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
    # Format 1: "INFO - val_acc = X.XX"
    # Format 2 (O*/EBR methods): "INFO - Initial val_acc = X.XX"
    # Format 3 (OGCc): "INFO - val acc = X.XX" (first; init test acc is not
    # logged for OGCc and must come from another method's log)
    m = re.search(r"INFO - (?:Initial )?val_acc = ([\-\d.]+)", content)
    if m:
        result["init_val_acc"] = float(m.group(1))
    else:
        m = re.search(r"INFO - val acc = ([\-\d.]+)", content)
        if m:
            result["init_val_acc"] = float(m.group(1))
    m = re.search(r"INFO - (?:Initial )?test_acc = ([\-\d.]+)", content)
    if m:
        result["init_test_acc"] = float(m.group(1))

    # Best performance (final validation accuracy)
    # Format 1: "INFO - Best performance = X.XX"
    # Format 2 (TMN): "INFO -     Accuracy Test: X.XXXX"
    # Format 3 (OGR/OGC): "INFO - After selection - Val Acc: X.XXXX, ..." (last)
    # Format 4 (EBR): "best accuracy = X.XX" (last)
    # Format 5 (OGCc): max over "INFO - val acc = X.XX" lines
    m = re.search(r"INFO - Best performance = ([\-\d.]+)", content)
    if m:
        result["best_val_acc"] = float(m.group(1))
    else:
        m = re.search(r"INFO -\s+Accuracy Test: ([\-\d.]+)", content)
        if m:
            result["best_val_acc"] = float(m.group(1))
        else:
            matches = re.findall(r"INFO - After selection - Val Acc: ([\-\d.]+)", content)
            if matches:
                result["best_val_acc"] = float(matches[-1])
            else:
                matches = re.findall(r"best accuracy = ([\-\d.]+)", content)
                if matches:
                    result["best_val_acc"] = float(matches[-1])
                else:
                    matches = re.findall(r"INFO - val acc = ([\-\d.]+)", content)
                    if matches:
                        result["best_val_acc"] = max(float(v) for v in matches)
                    else:
                        # Truncated logs without a summary line: best val acc
                        # over the per-iteration new/sel values.
                        matches = re.findall(r"INFO - (?:new|sel)_val_acc = ([\-\d.]+)", content)
                        if matches:
                            result["best_val_acc"] = max(float(v) for v in matches)

    # final_test_acc — tried in order:
    # Format 1: "INFO - final_test_acc = X.XX"
    # Format 2 (plain, e.g. CGC logs truncated mid-iteration): "final_test_acc = X.XX"
    # Format 3 (TMN): "rf final_test_acc = X.XX" or "rf final_test_acc_rf = X.XX"
    # Format 4 (O*/EBR methods): "INFO - final_test_acc_rf = X.XX" (last)
    # Format 5 (OGR/OGC): "INFO - After selection - Val Acc: ..., Test Acc: X.XX" (last)
    # Format 6 (OGCc): "final RandomForest Val: XX.XX | Test: XX.XX" (percent)
    # Format 7 (OGCc): "Val: X.XX | Test: X.XX" (last)
    m = re.search(r"INFO - final_test_acc = ([\-\d.]+)", content)
    if m:
        result["final_test_acc"] = float(m.group(1))
    if "final_test_acc" not in result:
        m = re.search(r"^final_test_acc = ([\-\d.]+)", content, re.MULTILINE)
        if m:
            result["final_test_acc"] = float(m.group(1))
    if "final_test_acc" not in result:
        m = re.search(r"^rf final_test_acc(?:_rf)? = ([\-\d.]+)", content, re.MULTILINE)
        if m:
            result["final_test_acc"] = float(m.group(1))
    if "final_test_acc" not in result:
        matches = re.findall(r"INFO - final_test_acc_rf = ([\-\d.]+)", content)
        if matches:
            result["final_test_acc"] = float(matches[-1])
    if "final_test_acc" not in result:
        matches = re.findall(r"INFO - After selection - Val Acc: [\-\d.]+, Test Acc: ([\-\d.]+)", content)
        if matches:
            result["final_test_acc"] = float(matches[-1])
    if "final_test_acc" not in result:
        m = re.search(r"final RandomForest Val: ([\-\d.]+) \| Test: ([\-\d.]+)", content)
        if m:
            # OGCc/CGCc print percents here, but normalize defensively:
            # accuracy fractions are always <= 1.
            result["final_test_acc"] = _normalize_acc(float(m.group(2)))
    if "final_test_acc" not in result:
        matches = re.findall(r"^Val: ([\-\d.]+) \| Test: ([\-\d.]+)", content, re.MULTILINE)
        if matches:
            # OGCc prints fractions (0.92), CGCc prints percents (77.0)
            result["best_val_acc"] = _normalize_acc(float(matches[-1][0]))
            result["final_test_acc"] = _normalize_acc(float(matches[-1][1]))
    if "final_test_acc" not in result:
        # Format 8 (GRFG one-shot):
        # "INFO:  [SEPARATE TEST] Acc on original is: X, Acc on generated is: Y"
        # Self-contained: the original accuracy serves as the initial value.
        # GRFG logs may use "Acc" (classification) or "1-RMSE" (regression).
        m = re.search(r"\[SEPARATE TEST\] (?:Acc|1-RMSE) on original is: ([\-\d.]+), (?:Acc|1-RMSE) on generated is: ([\-\d.]+)", content)
        if m:
            result.setdefault("init_test_acc", float(m.group(1)))
            result["final_test_acc"] = float(m.group(2))
        # GRFG also logs AutoGluon one-shot results:
        # "INFO:  [SEPARATE TEST AG] Acc on original is: X, Acc on generated is: Y"
        m_ag = re.search(r"\[SEPARATE TEST AG\] (?:Acc|1-RMSE) on original is: ([\-\d.]+), (?:Acc|1-RMSE) on generated is: ([\-\d.]+)", content)
        if m_ag:
            result["ag_change"] = float(m_ag.group(2)) - float(m_ag.group(1))
    if "final_test_acc" not in result and re.search(r"INFO - ag_acc = ", content):
        # Format 9 (AutoFeat one-shot): the single val_acc/test_acc pair is the
        # FINAL result. The initial accuracy is not logged; it must come from
        # another method's log on the same (dataset, seed).
        if "init_test_acc" in result:
            result["final_test_acc"] = result.pop("init_test_acc")
        if "init_val_acc" in result:
            result["best_val_acc"] = result.pop("init_val_acc")

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
    # Format 3 (O*/EBR methods): "INFO - final_test_acc_ag = X.XX" (last)
    # Format 4 (OGCc): "final ag test = X.XX"
    # Format 5 (Autofeat): "INFO - ag_acc = X.XX"
    # Format 6 (GRFG): "[AG] X.XX" (already a change, often in pp)
    m = re.search(r"^final_test_acc_ag = ([\-\d.]+)", content, re.MULTILINE)
    if m:
        result["final_test_acc_ag"] = float(m.group(1))
    else:
        m = re.search(r"^ag final_test_acc = ([\-\d.]+)", content, re.MULTILINE)
        if m:
            result["final_test_acc_ag"] = float(m.group(1))
        else:
            matches = re.findall(r"INFO - final_test_acc_ag = ([\-\d.]+)", content)
            if matches:
                result["final_test_acc_ag"] = float(matches[-1])
            else:
                m = re.search(r"^final ag(?: test)? = ([\-\d.]+)", content, re.MULTILINE)
                if m:
                    # OGCc logs mix fractions (0.77) and percents (75.5)
                    result["final_test_acc_ag"] = _normalize_acc(float(m.group(1)))
                else:
                    m = re.search(r"INFO - ag_acc = ([\-\d.]+)", content)
                    if m:
                        result["final_test_acc_ag"] = float(m.group(1))
                    else:
                        m = re.search(r"\[AG\]\s*([\-\d.]+)", content)
                        if m:
                            ag_change = float(m.group(1))
                            # GRFG prints AG change in percentage points (e.g. -4.0, +1.0)
                            if abs(ag_change) >= 1.0:
                                ag_change = ag_change / 100.0
                            result["ag_change"] = ag_change

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


def mean_std(values):
    """Return (mean, sample std) of a list of values. std is 0.0 when n < 2."""
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
    else:
        var = 0.0
    return mean, math.sqrt(var)


def iter_method_files(directory, methods, datasets, seeds):
    """Yield (method, filepath, dataset, seed) for all matching logs."""
    for method in methods:
        mdir = os.path.join(directory, method)
        if not os.path.isdir(mdir):
            print(f"WARNING: method directory not found, skipped: {mdir}")
            continue
        files = find_log_files(mdir, datasets, seeds)
        for fpath in files:
            fname = os.path.basename(fpath)
            ds = next(d for d in datasets if fname.startswith(d + "_"))
            seed = next(s for s in seeds if fname.endswith(f"_{s}.log"))
            yield method, fpath, ds, seed


def build_init_references(directory, method_groups, datasets, seeds):
    """
    Build reference maps (dataset, seed) -> initial val/test accuracy by
    scanning the logs of all method groups. Methods that do not log the
    initial accuracy (e.g., EBR*) use these references.

    Most methods share the same initial model for a given (dataset, seed),
    but occasional outliers exist (e.g., CGC logs show slightly different
    splits). To be robust, the reference is the MEDIAN over all methods
    that log the value, not the first one found.
    """
    import statistics
    test_samples = {}
    val_samples = {}
    for methods in method_groups:
        for _, fpath, ds, seed in iter_method_files(directory, methods, datasets, seeds):
            parsed = parse_log(fpath)
            key = (ds, seed)
            if parsed.get("init_test_acc") is not None:
                test_samples.setdefault(key, []).append(parsed["init_test_acc"])
            if parsed.get("init_val_acc") is not None:
                val_samples.setdefault(key, []).append(parsed["init_val_acc"])
    init_test_ref = {k: statistics.median(v) for k, v in test_samples.items()}
    init_val_ref = {k: statistics.median(v) for k, v in val_samples.items()}
    return init_test_ref, init_val_ref


def analyze_files(files, verbose=True, dataset=None, init_test_ref=None, init_val_ref=None):
    """Parse log files and collect per-file metrics.

    Returns a dict with metric lists (val/test/ag changes, tokens, times),
    lists of files with missing data, and the file count.

    When init_test_ref/init_val_ref are given (reference maps keyed by
    (dataset, seed)), files that do not log the initial accuracy (e.g.,
    EBR*) fall back to the reference value.
    """
    val_changes = []
    test_changes = []
    tokens = []
    times = []
    ag_changes = []
    val_missing = []
    test_missing = []
    ag_missing = []

    for fpath in files:
        r = parse_log(fpath)
        fname = os.path.basename(fpath)

        # Handle parse error case: set all acc improvements to 0
        if r.get("parse_error"):
            if verbose:
                print(f"--- {fname} ---")
                print(f"  ERROR: No columns to parse from file - setting all acc changes to 0\n")
            val_changes.append(0)
            test_changes.append(0)
            ag_changes.append(0)
            continue

        init_val = r.get("init_val_acc")
        best_val = r.get("best_val_acc")
        init_test = r.get("init_test_acc")
        final_test = r.get("final_test_acc")
        final_ag = r.get("final_test_acc_ag")
        data_name = r.get("data_name")
        seed = r.get("seed")

        # Fall back to reference initial accuracies (median across methods)
        if dataset is not None and (init_test is None or init_val is None):
            seed_key = seed
            if seed_key is None:
                m_seed = re.search(r"_(\d+)\.log$", fname)
                seed_key = int(m_seed.group(1)) if m_seed else None
            if seed_key is not None:
                if init_test is None and init_test_ref is not None:
                    init_test = init_test_ref.get((dataset, seed_key))
                if init_val is None and init_val_ref is not None:
                    init_val = init_val_ref.get((dataset, seed_key))

        if verbose:
            print(f"--- {fname} ---")
        if init_val is not None and best_val is not None:
            delta_val = best_val - init_val
            val_changes.append(delta_val)
            if verbose:
                print(f"  Val acc: {init_val:.6f} -> {best_val:.6f}  (Δ={delta_val:+.6f})")
        else:
            val_missing.append(fname)
            if verbose:
                print(f"  Val acc: MISSING DATA")

        if init_test is not None and final_test is not None:
            delta_test = final_test - init_test
            test_changes.append(delta_test)
            if verbose:
                print(f"  Test acc: {init_test:.6f} -> {final_test:.6f}  (Δ={delta_test:+.6f})")
        else:
            test_missing.append(fname)
            if verbose:
                print(f"  Test acc: MISSING DATA")

        if r.get("total_tokens") is not None:
            tokens.append(r["total_tokens"])
            if verbose:
                print(f"  Tokens: {r['total_tokens']:.0f}")

        if r.get("total_time") is not None:
            times.append(r["total_time"])
            if verbose:
                print(f"  Time: {r['total_time']:.2f}s")

        if final_ag is not None:
            ag_key = (data_name, seed) if data_name and seed is not None else None
            ag_init = AG_INIT.get(ag_key) if ag_key else None
            if ag_init is not None:
                delta_ag = final_ag - ag_init
                ag_changes.append(delta_ag)
                if verbose:
                    print(f"  AG acc: {ag_init:.6f} -> {final_ag:.6f}  (Δ={delta_ag:+.6f})")
            elif verbose:
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
                if verbose:
                    print(f"  AG acc: MISSING (only final_test_acc_ag missing, treating as Δ=0)")
            else:
                ag_missing.append(fname)
                if verbose:
                    print(f"  AG acc: NOT AVAILABLE")
        if verbose:
            print()

    return {
        "n_files": len(files),
        "val": val_changes, "test": test_changes, "ag": ag_changes,
        "tokens": tokens, "times": times,
        "val_missing": val_missing, "test_missing": test_missing,
        "ag_missing": ag_missing,
    }


def trimmed_stats(values):
    """Apply the trimmed-mean rule (lowest removed when n > 1) and return
    (mean, std, n). Returns None for an empty list."""
    if not values:
        return None
    if len(values) > 1:
        values = sorted(values)[1:]  # Remove lowest
    m, s = mean_std(values)
    return m, s, len(values)


def print_detailed_summary(stats):
    """Print the per-metric summary for single method/dataset mode."""
    print("=" * 50)
    print(f"Files processed: {stats['n_files']}")
    for label, key in [("val acc", "val"), ("test acc", "test"), ("AG acc", "ag")]:
        values = stats[key]
        missing = stats[f"{key}_missing"]
        if values:
            m, s, n = trimmed_stats(values)
            note = ", lowest removed" if len(values) > 1 else ""
            print(f"Mean {label} change:  {m:+.6f}  (std={s:.6f}, n={n}{note})")
            print(f"Max {label} change:   {max(values):+.6f}")
        if missing:
            print(f"  !! {label.capitalize()} missing in {len(missing)} file(s): {', '.join(missing)}")
        if key == "test":
            if stats["tokens"]:
                m, s = mean_std(stats["tokens"])
                print(f"Avg token cost:       {m:.0f}  (std={s:.0f}, n={len(stats['tokens'])})")
            if stats["times"]:
                m, s = mean_std(stats["times"])
                print(f"Avg time used:        {m:.2f}s  (std={s:.2f}s, n={len(stats['times'])})")


def _fmt_cell(stats_tuple):
    """Format a (mean, std, n) tuple as a table cell."""
    if stats_tuple is None:
        return "N/A"
    m, s, n = stats_tuple
    return f"{m:+.4f}±{s:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract and summarize log file metrics.",
        epilog="Multi mode: pass --datasets (plural) for several datasets and/or "
               "--methods (comma-separated) to treat DIRECTORY as a parent directory "
               "of method subdirectories; a summary table is printed.")
    parser.add_argument("directory", help="Directory containing log files "
                        "(or method subdirectories when --methods is given)")
    parser.add_argument("--dataset", help="Dataset name prefix (e.g., credit-g)")
    parser.add_argument("--datasets", nargs="+",
                        help="Multiple dataset name prefixes (enables table mode)")
    parser.add_argument("--methods",
                        help="Comma-separated method subdirectory names (enables multi-method mode)")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Random seeds to include (e.g., 1 2 3)")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ([args.dataset] if args.dataset else None)
    if not datasets:
        parser.error("one of --dataset or --datasets is required")

    methods = [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None

    # Single method + single dataset: detailed per-file output
    if methods is None and len(datasets) == 1:
        files = find_log_files(args.directory, datasets, args.seeds)
        if not files:
            print("No matching log files found.")
            return
        print_detailed_summary(analyze_files(files, verbose=True))
        return

    # Table mode: rows are (method, dataset) pairs
    rows = []  # (method, dataset, stats)
    if methods is None:
        method_dirs = [(os.path.basename(os.path.normpath(args.directory)), args.directory)]
    else:
        method_dirs = []
        for m in methods:
            mdir = os.path.join(args.directory, m)
            if not os.path.isdir(mdir):
                print(f"WARNING: method directory not found, skipped: {mdir}")
                continue
            method_dirs.append((m, mdir))

    # Reference initial accuracies (cross-method median) for logs that do not
    # record them (e.g., EBR*), built from the selected methods' logs.
    init_test_ref, init_val_ref = {}, {}
    by_parent = {}
    for m, mdir in method_dirs:
        norm = os.path.normpath(mdir)
        by_parent.setdefault(os.path.dirname(norm), []).append(os.path.basename(norm))
    for parent, names in by_parent.items():
        itr, ivr = build_init_references(parent, [names], datasets, args.seeds)
        init_test_ref.update(itr)
        init_val_ref.update(ivr)

    for method, mdir in method_dirs:
        for ds in datasets:
            files = find_log_files(mdir, [ds], args.seeds)
            if not files:
                rows.append((method, ds, None))
                continue
            rows.append((method, ds, analyze_files(
                files, verbose=False, dataset=ds,
                init_test_ref=init_test_ref, init_val_ref=init_val_ref)))

    show_method = methods is not None
    header = f"{'method':<10}" if show_method else ""
    header += f"{'dataset':<18}{'VG (mean±std)':>20}{'TG (mean±std)':>20}{'AG (mean±std)':>20}{'#files':>7}"
    print(header)
    print("-" * len(header))
    warnings = []
    for method, ds, stats in rows:
        line = f"{method:<10}" if show_method else ""
        if stats is None:
            line += f"{ds:<18}{'no logs':>20}{'':>20}{'':>20}{0:>7}"
            warnings.append(f"{method}/{ds}: no matching logs")
        else:
            line += f"{ds:<18}{_fmt_cell(trimmed_stats(stats['val'])):>20}" \
                    f"{_fmt_cell(trimmed_stats(stats['test'])):>20}" \
                    f"{_fmt_cell(trimmed_stats(stats['ag'])):>20}{stats['n_files']:>7}"
            for key, label in [("val", "VG"), ("test", "TG"), ("ag", "AG")]:
                for fname in stats[f"{key}_missing"]:
                    warnings.append(f"{method}/{ds}: {label} missing in {fname}")
        print(line)

    print()
    print("Note: mean±std computed after removing the lowest value (trimmed, n shown in")
    print("detailed mode); N/A = no usable data. AG requires AG_INIT entries.")
    if warnings:
        print()
        print("Warnings:")
        for w in warnings:
            print(f"  !! {w}")


if __name__ == "__main__":
    main()

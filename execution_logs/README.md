# Log Analysis Tools

To facilitate analysis without incurring over two weeks of runtime and $700+ in API costs, this directory provides 4,000+ high-quality execution logs for the various LATTE methods within the LATTEBench paper. We have filtered out logs that failed within short turns. All observations presented in the paper can be derived from these logs.

Owing to historical legacy issues accumulated during LATTEBench's iterative development, inconsistencies exist in both the content and format of these execution logs. We are committed to assisting researchers in leveraging these logs for LLM behavior analysis and model training; therefore, we provide a suite of Python scripts designed for automated analysis.

## Tool 1: `extract_log.py`

Extracts lines containing accuracy and token usage keywords from a set of numbered log files (e.g., `xxx_1.log` through `xxx_6.log`) and writes them to a single output file.

### Usage

```bash
python extract_log.py <base_filename> [-s START] [-e END] [-o OUTPUT]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `base_filename` | Base path of log files (e.g., `Main_Results/CGN/vehicle_CoT_gpt-4o_3`) | required |
| `-s`, `--start` | Starting file index | `1` |
| `-e`, `--end` | Ending file index | `6` |
| `-o`, `--output` | Output filename | `extracted_data.log` |

### Keyword Selection

Keywords are automatically chosen based on the method name extracted from `base_filename`:

| Method | Val Acc Keyword | Token Keyword |
|---|---|---|
| OGC, OGR | `Val Acc:` | `Total token usage =` |
| T* (TMN, TPO, etc.) | `val_acc =` | `Total tokens consumed in this batch:` |
| Others | `val_acc =` | `Total token usage =` |

### Example

```bash
python extract_log.py Main_Results/CGN/vehicle_CoT_gpt-4o_3 -s 1 -e 6 -o extracted_data.log
```

## Tool 2: `extract_log_unified.py`

Processes the extracted log file from Stage 1 and computes:

1. **Monotonic-max accuracy** per iteration (cumulative best accuracy across iterations)
2. **Token usage** per iteration

Both are averaged across all files (runs), with shorter runs padded using their last value. This can be used to plot the performance gain vs. token cost curve.

### Usage

```bash
python extract_log_unified.py [-i INPUT] [-p PATTERN] [-l LENGTH]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `-i`, `--input` | Input log filename | `extracted_data.log` |
| `-p`, `--pattern` | Log format pattern (`pattern1`, `pattern2`, `pattern3`, `auto`) | `auto` |
| `-l`, `--length` | Max data points per file | `20` |

### Supported Patterns

**Pattern 1** (TMN/T* methods) - Token appears before accuracy lines:
```
Total tokens consumed in this batch: 5122
new_val_acc = 0.7396
sel_val_acc = 0.7218
```

**Pattern 2** (CGN and similar methods) - Accuracy appears before token line:
```
new_val_acc = 0.7337
sel_val_acc = 0.7337
Total token usage = 2707
```

**Pattern 3** (OGR/OGC methods) - Turn-based dialogue format:
```
Turn 1 Val Acc: 0.7337, Test Acc: 0.7647
...
Turn 10 Val Acc: 0.7574, Test Acc: 0.7471
Best dialogue result - Val Acc: 0.7574, Test Acc: 0.7765
After selection - Val Acc: 0.7396, Test Acc: 0.7353
Total token usage = 26395
```

### Examples

```bash
# Auto-detect pattern
python extract_log_unified.py -i extracted_data.log

# Specify pattern explicitly
python extract_log_unified.py -i extracted_data.log -p pattern1

# Limit to 10 data points per file
python extract_log_unified.py -i extracted_data.log -l 10
```

### Typical Workflow

```bash
# 1. Extract relevant lines from raw log files
python extract_log.py Main_Results/CGN/vehicle_CoT_gpt-4o_3 -o extracted_data.log

# 2. Process extracted data (auto-detects pattern)
python extract_log_unified.py -i extracted_data.log
```

## Tool 3 `extract.py`

Parses raw log files for a single dataset across multiple seeds, extracts per-file metrics (val/test accuracy changes, AG accuracy changes, token usage, time), and outputs a summary with averaged results. Missing or unavailable data will be reported in the summary.

### Usage

```bash
python extract.py <directory> --dataset <name> --seeds <seed1> [seed2 ...]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `directory` | Directory containing log files | required |
| `--dataset` | Dataset name prefix (e.g., `credit-g`) | required |
| `--seeds` | Random seeds to include (e.g., `1 2 3 4 5 6`) | required |

### Metrics Extracted

- **Val acc change**: `best_val_acc - init_val_acc`
- **Test acc change**: `final_test_acc - init_test_acc`
- **AG acc change**: `final_test_acc_ag - AG_INIT[(dataset, seed)]` (initial AG values are specified in the script)
- **Token usage** and **time used**

### Summary Behavior

- When there are more than one results, the **lowest value is removed** before averaging (trimmed mean).
- If any files have missing or unavailable data, the summary will list the affected files. In this case, it is **highly recommended** to **manually** inspect the specific results in the log and record them, or delete the corresponding log and **re-run the bench**.

### Example

```bash
python extract.py Main_Results/CGC/ --dataset credit-g --seeds 1 2 3 4 5 6
```

##Tool 4 `check_success_rate.py`

Calculates round-level success rates from log files by detecting warnings/errors in each round. Supports three different log formats based on folder naming conventions.

### Usage

```bash
python check_success_rate.py <directory> --datasets <name1> [name2 ...] --seeds <seed1> [seed2 ...]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `directory` | Directory containing log files | required |
| `--datasets` | Dataset name prefixes to include (e.g., `credit-g kc1`) | required |
| `--seeds` | Random seeds to include (e.g., `1 2 3`) | required |

### Log Format Detection

The script automatically detects the log format based on the parent folder name:

| Folder Prefix | Mode | Round Definition |
|---|---|---|
| `O*` (e.g., `OGC`) | OPRO | Each iteration contains multiple dialogue turns. An iteration succeeds if ANY turn has no error. |
| `TM*` (e.g., `TMN`) | ToT | Each segment from "LLM Output:" to "Success Operators:" is a round. |
| Others | Default | Each iteration marker (`=== Iteration X/Y ===`) is a single round. |

### Error Detection

A round is considered to have an error if any of the following appear in its content:
- `Warning` (exact case)
- `warnings` (case-insensitive)
- `警告` (Chinese for "warning")

### Output

For each matching log file:
- Mode (OPRO/ToT/Default)
- Total rounds
- Rounds without errors
- Rounds with errors
- Success rate

Plus an overall summary across all files.

### Example

```bash
python check_success_rate.py Main_Results/CGN --datasets kc1 qsar-biodeg electricity nomao socmob bike_sharing cpu_small diamonds wine_quality heart-h credit-approval credit-g --seeds 1 2 3 4 5 6
```

<details>
<summary>View Example</summary>

```
--- bike_sharing_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- bike_sharing_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- bike_sharing_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- cpu_small_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- cpu_small_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 6
  Rounds with errors: 4
  Success rate: 0.6000 (6/10)

--- cpu_small_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- credit-approval_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 10
  Rounds with errors: 0
  Success rate: 1.0000 (10/10)

--- credit-approval_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 10
  Rounds with errors: 0
  Success rate: 1.0000 (10/10)

--- credit-approval_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- credit-approval_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- credit-approval_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- credit-approval_CoT_gpt-4o_3_6.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- credit-g_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- credit-g_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- credit-g_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- credit-g_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 6
  Rounds with errors: 4
  Success rate: 0.6000 (6/10)

--- credit-g_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- credit-g_CoT_gpt-4o_3_6.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- diamonds_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- diamonds_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- diamonds_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- electricity_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 10
  Rounds with errors: 0
  Success rate: 1.0000 (10/10)

--- electricity_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- electricity_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- electricity_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 6
  Rounds with errors: 4
  Success rate: 0.6000 (6/10)

--- electricity_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- heart-h_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- heart-h_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 5
  Rounds with errors: 5
  Success rate: 0.5000 (5/10)

--- heart-h_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- heart-h_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- heart-h_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- kc1_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- kc1_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 7
  Rounds with errors: 3
  Success rate: 0.7000 (7/10)

--- kc1_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 6
  Rounds with errors: 4
  Success rate: 0.6000 (6/10)

--- kc1_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- kc1_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- kc1_CoT_gpt-4o_3_6.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- nomao_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 5
  Rounds with errors: 5
  Success rate: 0.5000 (5/10)

--- nomao_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- qsar-biodeg_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- qsar-biodeg_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- qsar-biodeg_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 6
  Rounds with errors: 4
  Success rate: 0.6000 (6/10)

--- qsar-biodeg_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- qsar-biodeg_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- qsar-biodeg_CoT_gpt-4o_3_6.log ---
  Total rounds: 10
  Rounds without errors: 10
  Rounds with errors: 0
  Success rate: 1.0000 (10/10)

--- socmob_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- socmob_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- socmob_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- socmob_CoT_gpt-4o_3_4.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- socmob_CoT_gpt-4o_3_5.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- wine_quality_CoT_gpt-4o_3_1.log ---
  Total rounds: 10
  Rounds without errors: 9
  Rounds with errors: 1
  Success rate: 0.9000 (9/10)

--- wine_quality_CoT_gpt-4o_3_2.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

--- wine_quality_CoT_gpt-4o_3_3.log ---
  Total rounds: 10
  Rounds without errors: 8
  Rounds with errors: 2
  Success rate: 0.8000 (8/10)

==================================================
Files processed: 53
Total rounds: 530
Total rounds without errors: 426
Overall success rate: 0.8038 (426/530)
```
</details>
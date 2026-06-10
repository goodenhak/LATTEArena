"""
bench.py - Comprehensive benchmark runner for LATTEBench framework.

Related scripts:
    - baseline_eval.py:     Evaluate original features (no FE) with RF / AutoGluon
    - baseline_autofeat.py: Traditional AutoFeat feature engineering baseline

Usage Examples:
    # 0. Evaluate baselines first
    python baseline_eval.py --datasets credit-g vehicle --seeds 1 2 3
    python baseline_autofeat.py --data_name credit-g --seed 1

    # Run all methods on all datasets
    python bench.py

    # Run specific methods
    python bench.py --methods CoT Critic OPRO

    # Run on specific datasets
    python bench.py --datasets credit-g vehicle

    # Run with specific seeds
    python bench.py --seeds 1 2 3

    # Run with specific model
    python bench.py --llm_model gpt-4o

    # Dry run (show what would be executed)
    python bench.py --dry_run

    # Force re-run (ignore existing logs)
    python bench.py --force

    # Run only CoT with top-3 feedback
    python bench.py --methods CoT --top 3

    # Run OPRO with custom dialogue turns
    python bench.py --methods OPRO --dialogue_turns 5
"""

import subprocess
import os
import argparse
import time
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional


# ============================================================
# Default Configurations
# ============================================================

# All available methods
ALL_METHODS = ['CoT', 'LtM', 'Critic', 'OPROc', 'OPRO', 'ECoT', 'Evo', 'ToT']

# All available datasets (first 9 classification, last 4 regression)
ALL_DATASETS = [
    'heart-h',
    'credit-g',
    'vehicle',
    'kc1',
    'socmob',
    'credit-approval',
    'qsar-biodeg',
    'nomao',
    'electricity',
    'bike-sharing',
    'wine-quality',
    'diamonds',
    'cpu-small',
]

# Regression datasets (task_type=0); all others are classification (task_type=1)
REGRESSION_DATASETS = {'bike-sharing', 'wine-quality', 'diamonds', 'cpu-small'}

# Default configuration
DEFAULT_CONFIG = {
    # General
    'llm_model': 'gpt-4o',
    'seeds': [1, 2, 3, 4, 5, 6],
    'output_format': 'NL',
    'task_type': 1,
    'metadata_cat': 3,
    'selector': 1,
    'enlarge_num': 3,
    'test_size': 0.2,
    'val_size': 0.2,

    # Iteration counts (per method)
    'iter': 10,

    # CoT specific
    'history': 1,
    'top': 0,

    # OPRO/OPROc specific
    'dialogue_turns': 10,

    # Evo/ECoT specific
    'ind_num': 90,
    'remove_time': 1,
    'update_time': 2,
    'random_sample': 0,

    # ToT specific
    'num_thoughts': 2,
    'max_steps': 5,
    'max_states': 1,
    'pruning_threshold': 0.003,
    'search_strategy': 'mcts',

    # Serialization
    'ensemble': 1,
    'sample_size': 4,
    'sample_method': 1,
    'demo_format': 0,
    'op_type': 2,

    # LLM
    'max_tokens': 8192,
    'temperature': 1.0,
}

# Method-specific output format recommendations
METHOD_OUTPUT_FORMATS = {
    'CoT': ['NL', 'cRPN', 'Code'],
    'LtM': ['NL', 'cRPN', 'Code'],
    'Critic': ['NL', 'cRPN', 'Code'],
    'OPROc': ['Code'],  # OPROc uses Code format internally
    'OPRO': ['NL', 'cRPN', 'Code'],
    'ECoT': ['cRPN'],  # ECoT only supports cRPN
    'Evo': ['NL', 'cRPN', 'Code'],
    'ToT': ['NL', 'cRPN', 'Code'],
}

# Method-specific iteration recommendations
METHOD_ITERATIONS = {
    'CoT': 10,
    'LtM': 10,
    'Critic': 10,
    'OPROc': 10,  # Each iteration has multiple dialogue turns
    'OPRO': 10,   # Each iteration has multiple dialogue turns
    'ECoT': 10,
    'Evo': 10,
    'ToT': 1,     # ToT uses max_steps (default 5), not iterations
}


# ============================================================
# Argument Parser
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='LATTEBench Benchmark Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Method and dataset selection
    parser.add_argument('--methods', nargs='+', default=ALL_METHODS,
                        choices=ALL_METHODS, help='Methods to run')
    parser.add_argument('--datasets', nargs='+', default=ALL_DATASETS,
                        help='Datasets to run on')
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_CONFIG['seeds'],
                        help='Random seeds')
    parser.add_argument('--output_formats', nargs='+', default=None,
                        choices=['NL', 'cRPN', 'Code', 'Rule'],
                        help='Output formats (if not specified, uses method-specific defaults)')

    # Model configuration
    parser.add_argument('--llm_model', type=str, default=DEFAULT_CONFIG['llm_model'],
                        help='LLM model name')
    parser.add_argument('--task_type', type=int, default=DEFAULT_CONFIG['task_type'],
                        help='Task type (1=classification, 0=regression)')
    parser.add_argument('--metadata_cat', type=int, default=DEFAULT_CONFIG['metadata_cat'],
                        help='Metadata category')
    parser.add_argument('--test_size', type=float, default=DEFAULT_CONFIG['test_size'],
                        help='Test set ratio passed to latte.py')
    parser.add_argument('--val_size', type=float, default=DEFAULT_CONFIG['val_size'],
                        help='Validation set ratio passed to latte.py')

    # Iteration control
    parser.add_argument('--iter', type=int, default=None,
                        help='Number of iterations (overrides method defaults)')

    # CoT specific
    parser.add_argument('--history', type=int, default=DEFAULT_CONFIG['history'],
                        help='Use history feedback (CoT)')
    parser.add_argument('--top', type=int, default=DEFAULT_CONFIG['top'],
                        help='Use Top-k feedback for CoT/LtM; 0 disables, >0 sets k')

    # OPRO/OPROc specific
    parser.add_argument('--dialogue_turns', type=int, default=DEFAULT_CONFIG['dialogue_turns'],
                        help='Dialogue turns (OPRO/OPROc)')

    # Evo/ECoT specific
    parser.add_argument('--ind_num', type=int, default=DEFAULT_CONFIG['ind_num'],
                        help='Individuals per island (Evo/ECoT)')
    parser.add_argument('--remove_time', type=int, default=DEFAULT_CONFIG['remove_time'],
                        help='Remove weak individuals frequency (Evo/ECoT)')
    parser.add_argument('--update_time', type=int, default=DEFAULT_CONFIG['update_time'],
                        help='Inter-island update frequency (Evo/ECoT)')
    parser.add_argument('--random_sample', type=int, default=DEFAULT_CONFIG['random_sample'],
                        help='Random sampling in collect_data (ECoT)')

    # ToT specific
    parser.add_argument('--num_thoughts', type=int, default=DEFAULT_CONFIG['num_thoughts'],
                        help='Number of thoughts (ToT)')
    parser.add_argument('--max_steps', type=int, default=DEFAULT_CONFIG['max_steps'],
                        help='Max search steps (ToT)')
    parser.add_argument('--max_states', type=int, default=DEFAULT_CONFIG['max_states'],
                        help='Max states to maintain (ToT)')
    parser.add_argument('--search_strategy', type=str, default=DEFAULT_CONFIG['search_strategy'],
                        choices=['mcts', 'greedy'],
                        help='ToT node selection strategy')

    # Other parameters
    parser.add_argument('--selector', type=int, default=DEFAULT_CONFIG['selector'],
                        help='Use feature selector')
    parser.add_argument('--enlarge_num', type=int, default=DEFAULT_CONFIG['enlarge_num'],
                        help='Feature enlargement number')
    parser.add_argument('--sample_size', type=int, default=DEFAULT_CONFIG['sample_size'],
                        help='Sample size')
    parser.add_argument('--max_tokens', type=int, default=DEFAULT_CONFIG['max_tokens'],
                        help='Max tokens for LLM')
    parser.add_argument('--temperature', type=float, default=DEFAULT_CONFIG['temperature'],
                        help='Temperature for LLM')

    # Execution control
    parser.add_argument('--log_path', type=str, default='./log',
                        help='Log directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--force', action='store_true',
                        help='Force re-run even if log exists')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--save_config', type=str, default=None,
                        help='Save configuration to JSON file')
    parser.add_argument('--load_config', type=str, default=None,
                        help='Load configuration from JSON file')

    return parser.parse_args()


# ============================================================
# Utility Functions
# ============================================================

def get_log_filename(method: str, data_name: str, output_format: str,
                     llm_model: str, metadata_cat: int, seed: int) -> str:
    """Generate log filename based on method and parameters."""
    return f"{data_name}_{method}_{llm_model}_{metadata_cat}_{seed}.log"


def get_output_formats_for_method(method: str, user_formats: Optional[List[str]]) -> List[str]:
    """Get output formats for a method."""
    if user_formats:
        # Filter to only valid formats for this method
        valid = METHOD_OUTPUT_FORMATS.get(method, ['NL'])
        return [f for f in user_formats if f in valid] or [valid[0]]
    return [METHOD_OUTPUT_FORMATS.get(method, ['NL'])[0]]


def get_iterations_for_method(method: str, user_iter: Optional[int]) -> int:
    """Get iteration count for a method."""
    if user_iter is not None:
        return user_iter
    return METHOD_ITERATIONS.get(method, 50)


def build_command(method: str, data_name: str, seed: int, output_format: str,
                  args: argparse.Namespace) -> List[str]:
    """Build command line arguments for latte.py."""
    # Auto-determine task_type: 0 for regression datasets, 1 for classification
    task_type = 0 if data_name in REGRESSION_DATASETS else 1

    cmd = [
        'python', 'latte.py',
        f'--method={method}',
        f'--data_name={data_name}',
        f'--seed={seed}',
        f'--output_format={output_format}',
        f'--llm_model={args.llm_model}',
        f'--task_type={task_type}',
        f'--metadata_cat={args.metadata_cat}',
        f'--selector={args.selector}',
        f'--enlarge_num={args.enlarge_num}',
        f'--sample_size={args.sample_size}',
        f'--max_tokens={args.max_tokens}',
        f'--temperature={args.temperature}',
        f'--test_size={args.test_size}',
        f'--val_size={args.val_size}',
        f'--log_path={args.log_path}',
    ]

    # Add iteration count
    iterations = get_iterations_for_method(method, args.iter)
    cmd.append(f'--iter={iterations}')

    # Method-specific parameters
    if method in ['CoT', 'LtM']:
        cmd.append(f'--history={args.history}')
        cmd.append(f'--top={args.top}')

    elif method == 'Critic':
        pass  # No special params

    elif method in ['OPRO', 'OPROc']:
        cmd.append(f'--dialogue_turns={args.dialogue_turns}')

    elif method == 'ECoT':
        cmd.append(f'--ind_num={args.ind_num}')
        cmd.append(f'--remove_time={args.remove_time}')
        cmd.append(f'--update_time={args.update_time}')
        cmd.append(f'--random_sample={args.random_sample}')
        cmd.append('--op_type=0')  # ECoT requires op_type=0

    elif method == 'Evo':
        cmd.append(f'--ind_num={args.ind_num}')
        cmd.append(f'--remove_time={args.remove_time}')
        cmd.append(f'--update_time={args.update_time}')

    elif method == 'ToT':
        cmd.append(f'--num_thoughts={args.num_thoughts}')
        cmd.append(f'--max_steps={args.max_steps}')
        cmd.append(f'--max_states={args.max_states}')
        cmd.append(f'--history={args.history}')
        cmd.append(f'--search_strategy={args.search_strategy}')

    return cmd


def save_config(args: argparse.Namespace, filepath: str):
    """Save configuration to JSON file."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def parse_log_file(log_filepath: str) -> Optional[Dict[str, float]]:
    """
    Parse a log file and extract key metrics.

    Returns a dict with:
        - total_time: Total time used (seconds)
        - total_tokens: Total token usage
        - best_val: Best performance (validation accuracy)
        - final_test_rf: Final test accuracy (RF)
        - final_test_ag: Final test accuracy (AutoGluon)

    Returns None if the log file doesn't exist or execution failed.
    """
    if not os.path.exists(log_filepath):
        return None

    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if execution completed successfully
        if "========== END ==========" not in content:
            return None

        metrics = {}

        # Extract Total time used
        match = re.search(r'Total time used = ([\d.]+) seconds', content)
        if match:
            metrics['total_time'] = float(match.group(1))

        # Extract Total token usage (get the last occurrence)
        matches = re.findall(r'Total token usage = (\d+)', content)
        if matches:
            metrics['total_tokens'] = int(matches[-1])

        # Extract Best performance
        match = re.search(r'Best performance = ([\d.]+)', content)
        if match:
            metrics['best_val'] = float(match.group(1))

        # Extract final_test_acc_rf
        match = re.search(r'final_test_acc_rf = ([\d.]+)', content)
        if match:
            metrics['final_test_rf'] = float(match.group(1))

        # Extract final_test_acc_ag
        match = re.search(r'final_test_acc_ag = ([\d.]+)', content)
        if match:
            metrics['final_test_ag'] = float(match.group(1))

        # Only return if we have at least some key metrics
        if metrics:
            return metrics
        return None

    except Exception as e:
        print(f"Error parsing log file {log_filepath}: {e}")
        return None


# Pattern matching ERROR/WARNING log levels and Warning: in message content
_ISSUE_PATTERN = re.compile(r' - ERROR - | - WARNING - |Warning:')

# Methods that use dialogue turns within iterations
_DIALOGUE_METHODS = {'OPRO', 'OPROc'}

# Methods that use transformation-based success counting
_TRANSFORMATION_METHODS = {'ECoT', 'Evo'}

# Methods that use generate thoughts markers (ToT)
_THOUGHT_METHODS = {'ToT'}


def parse_log_success_rate(log_filepath: str, method: str) -> Optional[Dict[str, Any]]:
    """
    Parse a log file to calculate per-iteration success rate.

    An iteration is considered successful if it contains no Error/Warning.
    For OPRO/OPROc (dialogue methods), an iteration is successful as long as
    at least one dialogue turn within it has no Error/Warning.
    For ECoT/Evo, success rate is based on "new transformation found" vs
    "no valid transformation" counts.
    For ToT, each "generate thoughts" block (LLM call) is counted.
    For CoT, Critic, etc., each iteration (LLM output) is counted.

    Returns a dict with:
        - total_iters: Total number of iterations/transformations
        - success_iters: Number of successful iterations/transformations
        - success_rate: success_iters / total_iters
    Returns None if the log file doesn't exist or execution failed.
    """
    if not os.path.exists(log_filepath):
        return None

    try:
        with open(log_filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        if "========== END ==========" not in content:
            return None

        # ECoT/Evo: count transformation success/failure
        if method in _TRANSFORMATION_METHODS:
            success_count = len(re.findall(r'new transformation found', content, re.IGNORECASE))
            failure_count = len(re.findall(r'no valid transformation', content, re.IGNORECASE))
            total_iters = success_count + failure_count
            success_iters = success_count

        # ToT: split by "---generate thoughts---" markers (each LLM call)
        elif method in _THOUGHT_METHODS:
            thought_splits = re.split(r'---generate thoughts---', content)

            if len(thought_splits) <= 1:
                # No thought markers: treat entire body as one block
                thought_blocks = [content]
            else:
                # First element is preamble before the first thought generation; skip it
                thought_blocks = thought_splits[1:]

            total_iters = len(thought_blocks)
            success_iters = 0

            for block in thought_blocks:
                if not _ISSUE_PATTERN.search(block):
                    success_iters += 1

        else:
            # Default: Split by iteration markers (CoT, Critic, etc.)
            iter_splits = re.split(r'={5,} Iteration \d+/\d+ ={5,}', content)

            if len(iter_splits) <= 1:
                # No iteration markers: treat entire body as one iteration
                iter_blocks = [content]
            else:
                # First element is preamble before the first iteration; skip it
                iter_blocks = iter_splits[1:]

            total_iters = len(iter_blocks)
            success_iters = 0

            for block in iter_blocks:
                if method in _DIALOGUE_METHODS:
                    # Split by dialogue turn markers
                    turn_splits = re.split(r'--- Dialogue Turn \d+/\d+ ---', block)
                    turn_blocks = turn_splits[1:] if len(turn_splits) > 1 else [block]

                    # Iteration succeeds if at least one turn has no Error/Warning
                    if any(not _ISSUE_PATTERN.search(turn) for turn in turn_blocks):
                        success_iters += 1
                else:
                    if not _ISSUE_PATTERN.search(block):
                        success_iters += 1

        success_rate = success_iters / total_iters if total_iters > 0 else 0.0

        return {
            'total_iters': total_iters,
            'success_iters': success_iters,
            'success_rate': success_rate
        }

    except Exception as e:
        print(f"Error parsing success rate from {log_filepath}: {e}")
        return None


# ============================================================
# Experiment Runner
# ============================================================

class BenchmarkRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.results = []
        self.skipped = []
        self.failed = []
        self.start_time = None

    def run(self):
        """Run all experiments."""
        self.start_time = time.time()
        total_experiments = self._count_experiments()
        current = 0

        print("=" * 70)
        print("LATTEBench Benchmark Runner")
        print("=" * 70)
        print(f"Methods: {self.args.methods}")
        print(f"Datasets: {self.args.datasets}")
        print(f"Seeds: {self.args.seeds}")
        print(f"LLM Model: {self.args.llm_model}")
        print(f"Total experiments: {total_experiments}")
        print("=" * 70)

        if self.args.dry_run:
            print("\n*** DRY RUN MODE - Commands will be shown but not executed ***\n")

        os.makedirs(self.args.log_path, exist_ok=True)

        for method in self.args.methods:
            output_formats = get_output_formats_for_method(method, self.args.output_formats)

            for output_format in output_formats:
                for data_name in self.args.datasets:
                    for seed in self.args.seeds:
                        current += 1
                        self._run_single_experiment(
                            method, data_name, seed, output_format,
                            current, total_experiments
                        )

        self._print_summary()

    def _count_experiments(self) -> int:
        """Count total number of experiments."""
        count = 0
        for method in self.args.methods:
            formats = get_output_formats_for_method(method, self.args.output_formats)
            count += len(formats) * len(self.args.datasets) * len(self.args.seeds)
        return count

    def _run_single_experiment(self, method: str, data_name: str, seed: int,
                                output_format: str, current: int, total: int):
        """Run a single experiment."""
        log_filename = get_log_filename(
            method, data_name, output_format,
            self.args.llm_model, self.args.metadata_cat, seed
        )
        log_filepath = os.path.join(self.args.log_path, log_filename)

        # Check if already completed
        if os.path.exists(log_filepath) and not self.args.force:
            print(f"\n[{current}/{total}] SKIP: {method} | {data_name} | {output_format} | seed={seed}")
            print(f"         Log exists: {log_filename}")
            self.skipped.append({
                'method': method,
                'data_name': data_name,
                'output_format': output_format,
                'seed': seed,
                'reason': 'log_exists'
            })
            return

        # Build command
        cmd = build_command(method, data_name, seed, output_format, self.args)

        print(f"\n[{current}/{total}] RUN: {method} | {data_name} | {output_format} | seed={seed}")

        if self.args.verbose or self.args.dry_run:
            print(f"         Command: {' '.join(cmd)}")

        if self.args.dry_run:
            return

        # Execute
        exp_start = time.time()
        try:
            result = subprocess.run(cmd, check=True, capture_output=not self.args.verbose)
            exp_time = time.time() - exp_start

            print(f"         SUCCESS ({exp_time:.1f}s)")
            self.results.append({
                'method': method,
                'data_name': data_name,
                'output_format': output_format,
                'seed': seed,
                'status': 'success',
                'time': exp_time
            })

        except subprocess.CalledProcessError as e:
            exp_time = time.time() - exp_start
            print(f"         FAILED ({exp_time:.1f}s): {e}")
            self.failed.append({
                'method': method,
                'data_name': data_name,
                'output_format': output_format,
                'seed': seed,
                'error': str(e),
                'time': exp_time
            })
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Printing summary...\n")
            self._print_summary()
            raise

    def _print_summary(self):
        """Print experiment summary with metrics parsed from log files."""
        total_time = time.time() - self.start_time if self.start_time else 0

        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"Total wall time: {total_time:.1f}s ({total_time/60:.1f}m)")
        print(f"Completed: {len(self.results)}")
        print(f"Skipped:   {len(self.skipped)}")
        print(f"Failed:    {len(self.failed)}")

        if self.failed:
            print("\nFailed experiments:")
            for f in self.failed:
                print(f"  - {f['method']} | {f['data_name']} | {f['output_format']} | seed={f['seed']}")
                print(f"    Error: {f['error']}")

        # Parse log files to get actual metrics
        all_experiments = self.results + self.skipped

        def _new_bucket():
            return {
                'time': [], 'tokens': [], 'val': [], 'test_rf': [], 'test_ag': [],
                'success_rate': [], 'total_iters': [], 'success_iters': [],
                'num_completed': 0
            }

        # {method: {dataset: {<bucket>}}}
        method_dataset_metrics = {}

        for exp in all_experiments:
            method = exp['method']
            data_name = exp['data_name']
            log_filename = get_log_filename(
                method, data_name, exp.get('output_format', 'NL'),
                self.args.llm_model, self.args.metadata_cat, exp['seed']
            )
            log_filepath = os.path.join(self.args.log_path, log_filename)
            metrics = parse_log_file(log_filepath)
            success_metrics = parse_log_success_rate(log_filepath, method)

            if metrics:
                if method not in method_dataset_metrics:
                    method_dataset_metrics[method] = {}
                if data_name not in method_dataset_metrics[method]:
                    method_dataset_metrics[method][data_name] = _new_bucket()

                bucket = method_dataset_metrics[method][data_name]
                bucket['num_completed'] += 1
                if 'total_time' in metrics:
                    bucket['time'].append(metrics['total_time'])
                if 'total_tokens' in metrics:
                    bucket['tokens'].append(metrics['total_tokens'])
                if 'best_val' in metrics:
                    bucket['val'].append(metrics['best_val'])
                if 'final_test_rf' in metrics:
                    bucket['test_rf'].append(metrics['final_test_rf'])
                if 'final_test_ag' in metrics:
                    bucket['test_ag'].append(metrics['final_test_ag'])

                # Add success rate metrics
                if success_metrics:
                    bucket['success_rate'].append(success_metrics['success_rate'])
                    bucket['total_iters'].append(success_metrics['total_iters'])
                    bucket['success_iters'].append(success_metrics['success_iters'])

        def _print_bucket(m, total_runs, indent="  "):
            """Print metrics for a single bucket."""
            if m['time']:
                avg_time = sum(m['time']) / len(m['time'])
                print(f"{indent}Avg Time:         {avg_time:.2f}s ({len(m['time'])}/{total_runs} valid)")
            if m['tokens']:
                avg_tokens = sum(m['tokens']) / len(m['tokens'])
                print(f"{indent}Avg Tokens:       {avg_tokens:.0f} ({len(m['tokens'])}/{total_runs} valid)")
            if m['val']:
                avg_val = sum(m['val']) / len(m['val'])
                print(f"{indent}Avg Val (Best):   {avg_val:.4f} ({len(m['val'])}/{total_runs} valid)")
            if m['test_rf']:
                avg_test_rf = sum(m['test_rf']) / len(m['test_rf'])
                print(f"{indent}Avg Test (RF):    {avg_test_rf:.4f} ({len(m['test_rf'])}/{total_runs} valid)")
            if m['test_ag']:
                avg_test_ag = sum(m['test_ag']) / len(m['test_ag'])
                print(f"{indent}Avg Test (AG):    {avg_test_ag:.4f} ({len(m['test_ag'])}/{total_runs} valid)")
            if m['success_rate']:
                avg_sr = sum(m['success_rate']) / len(m['success_rate'])
                total_si = sum(m['success_iters'])
                total_ti = sum(m['total_iters'])
                overall_sr = total_si / total_ti if total_ti > 0 else 0.0
                print(f"{indent}Avg Success Rate: {avg_sr:.2%} (overall {total_si}/{total_ti}={overall_sr:.2%}, {len(m['success_rate'])} runs)")

        def _bucket_stats(m):
            """Compute summary stats dict from a bucket."""
            total_si = sum(m['success_iters']) if m['success_iters'] else 0
            total_ti = sum(m['total_iters']) if m['total_iters'] else 0
            return {
                'num_completed': m['num_completed'],
                'num_valid_test_rf': len(m['test_rf']),
                'num_valid_test_ag': len(m['test_ag']),
                'avg_time': sum(m['time']) / len(m['time']) if m['time'] else None,
                'avg_tokens': sum(m['tokens']) / len(m['tokens']) if m['tokens'] else None,
                'avg_val': sum(m['val']) / len(m['val']) if m['val'] else None,
                'avg_test_rf': sum(m['test_rf']) / len(m['test_rf']) if m['test_rf'] else None,
                'avg_test_ag': sum(m['test_ag']) / len(m['test_ag']) if m['test_ag'] else None,
                'avg_success_rate': sum(m['success_rate']) / len(m['success_rate']) if m['success_rate'] else None,
                'overall_success_iters': total_si,
                'overall_total_iters': total_ti,
                'overall_success_rate': total_si / total_ti if total_ti > 0 else None,
                'raw_time': m['time'],
                'raw_tokens': m['tokens'],
                'raw_val': m['val'],
                'raw_test_rf': m['test_rf'],
                'raw_test_ag': m['test_ag'],
                'raw_success_rate': m['success_rate'],
            }

        if method_dataset_metrics:
            print("\n" + "-" * 70)
            print("METRICS BY METHOD (parsed from log files)")
            print("-" * 70)

            for method in sorted(method_dataset_metrics.keys()):
                datasets = method_dataset_metrics[method]
                print(f"\n[{method}]")

                # Per-dataset breakdown
                for data_name in sorted(datasets.keys()):
                    m = datasets[data_name]
                    ds_runs = m['num_completed']
                    print(f"  <{data_name}> ({ds_runs} runs)")
                    _print_bucket(m, ds_runs, indent="    ")

        print("\n" + "=" * 70)

        # Compute summary statistics for JSON output
        summary_stats = {}
        for method, datasets in method_dataset_metrics.items():
            summary_stats[method] = {}
            for data_name, m in datasets.items():
                summary_stats[method][data_name] = _bucket_stats(m)

        # Save results to JSON
        results_file = os.path.join(self.args.log_path,
                                     f"bench_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_file, 'w') as f:
            json.dump({
                'config': vars(self.args),
                'results': self.results,
                'skipped': self.skipped,
                'failed': self.failed,
                'total_wall_time': total_time,
                'summary_stats': summary_stats
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")


# ============================================================
# Main
# ============================================================

def main():
    args = parse_arguments()

    # Load config from file if specified
    if args.load_config:
        loaded = load_config(args.load_config)
        for key, value in loaded.items():
            if hasattr(args, key) and key not in ['timestamp']:
                setattr(args, key, value)
        print(f"Configuration loaded from {args.load_config}")

    # Save config to file if specified
    if args.save_config:
        save_config(args, args.save_config)
        if args.dry_run:
            return

    runner = BenchmarkRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

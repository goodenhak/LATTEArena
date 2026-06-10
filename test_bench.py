#!/usr/bin/env python3
"""
test_bench.py - Quick functional test for bench.py

This script runs a minimal benchmark to verify:
1. Experiment execution works
2. Log file generation works
3. Log parsing and metrics extraction works
4. Success rate calculation works

Usage:
    python test_bench.py                    # Run with default test config
    python test_bench.py --dry_run          # Show commands without executing
    python test_bench.py --skip_run         # Skip running, only parse existing logs
    python test_bench.py --methods CoT      # Test specific method(s)
"""

import subprocess
import os
import sys
import argparse
import shutil
from datetime import datetime


# =============================================================================
# Test Configuration (minimal for quick testing)
# =============================================================================

TEST_CONFIG = {
    # Test all methods
    'methods': ['CoT', 'LtM', 'Critic', 'OPRO', 'OPROc', 'ToT', 'ECoT'],

    # Use only 1 small dataset
    'datasets': ['credit-g'],

    # Use only 1 seed
    'seeds': [1],

    # Use minimal iterations
    'iter': 2,

    # OPRO specific: minimal dialogue turns
    'dialogue_turns': 2,

    # ToT specific: minimal steps and thoughts
    'max_steps': 2,
    'num_thoughts': 2,
    'max_states': 1,

    # Use a separate log directory for tests
    'log_path': './log_test',

    # Other settings
    'llm_model': 'gpt-4o',
    'metadata_cat': 3,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Test bench.py functionality')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show commands without executing')
    parser.add_argument('--skip_run', action='store_true',
                        help='Skip running experiments, only test log parsing')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Override test methods')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Override test datasets')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Override test seeds')
    parser.add_argument('--iter', type=int, default=None,
                        help='Override iteration count')
    parser.add_argument('--keep_logs', action='store_true',
                        help='Keep test logs after completion')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    return parser.parse_args()


def build_bench_command(config, dry_run=False, verbose=False):
    """Build the bench.py command with test configuration."""
    cmd = [
        'python', 'bench.py',
        '--methods', *config['methods'],
        '--datasets', *config['datasets'],
        '--seeds', *[str(s) for s in config['seeds']],
        f'--iter={config["iter"]}',
        f'--dialogue_turns={config["dialogue_turns"]}',
        f'--max_steps={config["max_steps"]}',
        f'--num_thoughts={config["num_thoughts"]}',
        f'--max_states={config["max_states"]}',
        f'--log_path={config["log_path"]}',
        f'--llm_model={config["llm_model"]}',
        f'--metadata_cat={config["metadata_cat"]}',
    ]

    if dry_run:
        cmd.append('--dry_run')
    if verbose:
        cmd.append('--verbose')

    return cmd


def run_bench_test(config, dry_run=False, verbose=False):
    """Run bench.py with test configuration."""
    cmd = build_bench_command(config, dry_run, verbose)

    print("\n" + "=" * 70)
    print("Running bench.py with test configuration")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: bench.py failed with return code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return False


def verify_log_files(config):
    """Verify that log files were created and contain expected content."""
    print("\n" + "=" * 70)
    print("Verifying log files")
    print("=" * 70)

    log_path = config['log_path']
    if not os.path.exists(log_path):
        print(f"ERROR: Log directory '{log_path}' does not exist")
        return False

    expected_logs = []
    for method in config['methods']:
        for dataset in config['datasets']:
            for seed in config['seeds']:
                log_name = f"{dataset}_{method}_{config['llm_model']}_{config['metadata_cat']}_{seed}.log"
                expected_logs.append(log_name)

    all_ok = True
    for log_name in expected_logs:
        log_file = os.path.join(log_path, log_name)
        if os.path.exists(log_file):
            # Check if log has END marker
            with open(log_file, 'r') as f:
                content = f.read()

            has_end = "========== END ==========" in content
            has_start = "========== START" in content

            status = "OK" if (has_start and has_end) else "INCOMPLETE"
            size = os.path.getsize(log_file)
            print(f"  [{status}] {log_name} ({size} bytes)")

            if not has_end:
                all_ok = False
        else:
            print(f"  [MISSING] {log_name}")
            all_ok = False

    # Check for bench results JSON
    json_files = [f for f in os.listdir(log_path) if f.startswith('bench_results_') and f.endswith('.json')]
    if json_files:
        print(f"\n  Found {len(json_files)} bench results file(s):")
        for jf in sorted(json_files):
            size = os.path.getsize(os.path.join(log_path, jf))
            print(f"    - {jf} ({size} bytes)")
    else:
        print("\n  WARNING: No bench results JSON file found")

    return all_ok


def test_log_parsing(config):
    """Test log parsing functions."""
    print("\n" + "=" * 70)
    print("Testing log parsing functions")
    print("=" * 70)

    # Import parsing functions
    sys.path.insert(0, '.')
    from bench import parse_log_file, parse_log_success_rate

    log_path = config['log_path']
    all_ok = True

    for method in config['methods']:
        for dataset in config['datasets']:
            for seed in config['seeds']:
                log_name = f"{dataset}_{method}_{config['llm_model']}_{config['metadata_cat']}_{seed}.log"
                log_file = os.path.join(log_path, log_name)

                print(f"\n  Parsing: {log_name}")

                # Test parse_log_file
                metrics = parse_log_file(log_file)
                if metrics:
                    print(f"    Metrics:")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            print(f"      {key}: {value:.4f}")
                        else:
                            print(f"      {key}: {value}")
                else:
                    print(f"    Metrics: None (parsing failed or incomplete)")
                    all_ok = False

                # Test parse_log_success_rate
                sr_result = parse_log_success_rate(log_file, method)
                if sr_result:
                    print(f"    Success Rate:")
                    print(f"      total_iters: {sr_result['total_iters']}")
                    print(f"      success_iters: {sr_result['success_iters']}")
                    print(f"      success_rate: {sr_result['success_rate']:.2%}")
                else:
                    print(f"    Success Rate: None (parsing failed or incomplete)")
                    all_ok = False

    return all_ok


def cleanup_test_logs(config):
    """Clean up test log directory."""
    log_path = config['log_path']
    if os.path.exists(log_path):
        print(f"\nCleaning up test logs in '{log_path}'...")
        shutil.rmtree(log_path)
        print("Done.")


def main():
    args = parse_args()

    # Build test config with overrides
    config = TEST_CONFIG.copy()
    if args.methods:
        config['methods'] = args.methods
    if args.datasets:
        config['datasets'] = args.datasets
    if args.seeds:
        config['seeds'] = args.seeds
    if args.iter:
        config['iter'] = args.iter

    print("=" * 70)
    print("BENCH.PY FUNCTIONAL TEST")
    print("=" * 70)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTest Configuration:")
    print(f"  Methods:        {config['methods']}")
    print(f"  Datasets:       {config['datasets']}")
    print(f"  Seeds:          {config['seeds']}")
    print(f"  Iterations:     {config['iter']}")
    print(f"  Dialogue turns: {config['dialogue_turns']}")
    print(f"  Max steps:      {config['max_steps']} (ToT)")
    print(f"  Num thoughts:   {config['num_thoughts']} (ToT)")
    print(f"  Max states:     {config['max_states']} (ToT)")
    print(f"  Log path:       {config['log_path']}")

    total_experiments = len(config['methods']) * len(config['datasets']) * len(config['seeds'])
    print(f"\n  Total experiments: {total_experiments}")

    success = True

    # Step 1: Run bench.py (unless --skip_run)
    if not args.skip_run:
        if not run_bench_test(config, args.dry_run, args.verbose):
            success = False
            if not args.dry_run:
                print("\nWARNING: bench.py execution had issues")
    else:
        print("\n[Skipping experiment run as requested]")

    # Step 2: Verify log files (unless dry run)
    if not args.dry_run:
        if not verify_log_files(config):
            success = False
            print("\nWARNING: Some log files are missing or incomplete")

        # Step 3: Test log parsing
        if not test_log_parsing(config):
            success = False
            print("\nWARNING: Some log parsing tests failed")

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    if args.dry_run:
        print("Dry run completed - no actual experiments were executed")
    elif success:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED - check output above for details")

    # Cleanup (unless --keep_logs)
    if not args.keep_logs and not args.dry_run and not args.skip_run:
        cleanup_test_logs(config)
    elif args.keep_logs:
        print(f"\nTest logs kept in: {config['log_path']}")

    print(f"\nTest finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

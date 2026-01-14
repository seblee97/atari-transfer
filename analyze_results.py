#!/usr/bin/env python3
"""
Analyze transfer learning results from Atari RL experiments.

This script parses experiment directories, extracts performance metrics,
and generates comparison analyses across different algorithms, game pairs,
and transfer learning configurations.
"""

import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
import csv


def parse_progress_csv(csv_path):
    """Parse a progress.csv file from stable-baselines3 training logs."""
    if not os.path.exists(csv_path):
        return None

    metrics = {
        'timesteps': [],
        'episodes': [],
        'mean_reward': [],
        'exploration_rate': [],
    }

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'time/total_timesteps' in row:
                    metrics['timesteps'].append(float(row['time/total_timesteps']))

                # Try both rollout/ep_rew_mean (training) and eval/mean_reward (evaluation)
                reward_value = None
                if 'rollout/ep_rew_mean' in row and row['rollout/ep_rew_mean']:
                    reward_value = float(row['rollout/ep_rew_mean'])
                elif 'eval/mean_reward' in row and row['eval/mean_reward']:
                    reward_value = float(row['eval/mean_reward'])

                if reward_value is not None:
                    metrics['mean_reward'].append(reward_value)

                if 'time/episodes' in row and row['time/episodes']:
                    metrics['episodes'].append(float(row['time/episodes']))
                if 'rollout/exploration_rate' in row and row['rollout/exploration_rate']:
                    metrics['exploration_rate'].append(float(row['rollout/exploration_rate']))
    except Exception as e:
        print(f"Warning: Error parsing {csv_path}: {e}")
        return None

    return metrics


def load_experiment_results(results_dir):
    """Load all experiment results from the results directory."""
    experiments = []

    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return experiments

    for exp_dir in os.listdir(results_dir):
        exp_path = os.path.join(results_dir, exp_dir)

        # Skip non-directories and special directories
        if not os.path.isdir(exp_path) or exp_dir in ['slurm_scripts', 'slurm_logs']:
            continue

        # Load experiment config
        config_path = os.path.join(exp_path, 'config.json')
        if not os.path.exists(config_path):
            continue

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Parse source training logs
        source_csv = os.path.join(exp_path, 'source_logs', 'progress.csv')
        source_metrics = parse_progress_csv(source_csv)

        # Parse target training logs
        target_csv = os.path.join(exp_path, 'target_logs', 'progress.csv')
        target_metrics = parse_progress_csv(target_csv)

        # Load results.json if it exists
        results_path = os.path.join(exp_path, 'results.json')
        results_info = {}
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results_info = json.load(f)

        experiments.append({
            'name': exp_dir,
            'config': config,
            'source_metrics': source_metrics,
            'target_metrics': target_metrics,
            'results_info': results_info,
            'path': exp_path,
        })

    return experiments


def compute_summary_stats(metrics):
    """Compute summary statistics from training metrics."""
    if not metrics or not metrics.get('mean_reward'):
        return None

    rewards = metrics['mean_reward']

    return {
        'final_reward': rewards[-1] if rewards else None,
        'max_reward': max(rewards) if rewards else None,
        'mean_reward': sum(rewards) / len(rewards) if rewards else None,
        'num_episodes': metrics['episodes'][-1] if metrics.get('episodes') else None,
        'total_timesteps': metrics['timesteps'][-1] if metrics.get('timesteps') else None,
    }


def analyze_transfer_performance(experiments):
    """Analyze transfer learning performance across experiments."""

    # Group experiments by algorithm
    by_algorithm = defaultdict(list)

    for exp in experiments:
        algorithm = exp['config'].get('algorithm', 'unknown')
        by_algorithm[algorithm].append(exp)

    print("\n" + "="*80)
    print("TRANSFER LEARNING PERFORMANCE ANALYSIS")
    print("="*80)

    # Analyze each algorithm
    for algorithm in sorted(by_algorithm.keys()):
        exps = by_algorithm[algorithm]
        print(f"\n{'='*80}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Total experiments: {len(exps)}")
        print(f"{'='*80}")

        # Collect transfer pairs
        transfer_results = []

        for exp in exps:
            config = exp['config']
            source_game = config.get('source_game', 'unknown')
            target_game = config.get('target_game', 'unknown')
            freeze_encoder = config.get('freeze_encoder', False)
            reinit_head = config.get('reinit_head', False)

            source_stats = compute_summary_stats(exp['source_metrics'])
            target_stats = compute_summary_stats(exp['target_metrics'])

            if source_stats and target_stats:
                transfer_results.append({
                    'source_game': source_game,
                    'target_game': target_game,
                    'freeze_encoder': freeze_encoder,
                    'reinit_head': reinit_head,
                    'source_final_reward': source_stats['final_reward'],
                    'source_max_reward': source_stats['max_reward'],
                    'target_final_reward': target_stats['final_reward'],
                    'target_max_reward': target_stats['max_reward'],
                    'source_timesteps': source_stats['total_timesteps'],
                    'target_timesteps': target_stats['total_timesteps'],
                    'name': exp['name'],
                })

        # Print results table
        if transfer_results:
            print(f"\n{' '*20}Source Game Performance -> Target Game Performance")
            print(f"{'-'*80}")
            print(f"{'Transfer Pair':<30} {'Config':<20} {'Source Final':<15} {'Target Final':<15}")
            print(f"{'-'*80}")

            for result in sorted(transfer_results, key=lambda x: (x['source_game'], x['target_game'])):
                pair = f"{result['source_game']} -> {result['target_game']}"
                config_str = ""
                if result['freeze_encoder']:
                    config_str += "freeze"
                if result['reinit_head']:
                    config_str += "+reinit" if config_str else "reinit"
                if not config_str:
                    config_str = "baseline"

                print(f"{pair:<30} {config_str:<20} {result['source_final_reward']:<15.2f} {result['target_final_reward']:<15.2f}")
        else:
            print("\nNo completed experiments with metrics found.")


def analyze_transfer_benefit(experiments):
    """Analyze transfer learning benefit by comparing against from-scratch baselines.

    The key insight: For each game, we have a 'from-scratch' baseline when it appears
    as the SOURCE game (no pre-training), and transfer results when it appears as the
    TARGET game (with pre-training from various source games).
    """

    print("\n" + "="*80)
    print("TRANSFER LEARNING BENEFIT ANALYSIS")
    print("="*80)
    print("\nComparing transfer performance vs from-scratch baseline")
    print("Baseline = performance when game is trained as source (no pre-training)")
    print("Transfer = performance when game is trained as target (with pre-training)")

    # Group by algorithm and game
    by_algo_game = defaultdict(lambda: defaultdict(lambda: {'baseline': [], 'transfers': []}))

    for exp in experiments:
        config = exp['config']
        algorithm = config.get('algorithm', 'unknown')
        source_game = config.get('source_game', 'unknown')
        target_game = config.get('target_game', 'unknown')

        source_stats = compute_summary_stats(exp['source_metrics'])
        target_stats = compute_summary_stats(exp['target_metrics'])

        # Source game performance = from-scratch baseline for that game
        if source_stats:
            by_algo_game[algorithm][source_game]['baseline'].append({
                'reward': source_stats['final_reward'],
                'max_reward': source_stats['max_reward'],
                'timesteps': source_stats['total_timesteps'],
            })

        # Target game performance = transfer learning result
        if target_stats:
            by_algo_game[algorithm][target_game]['transfers'].append({
                'reward': target_stats['final_reward'],
                'max_reward': target_stats['max_reward'],
                'timesteps': target_stats['total_timesteps'],
                'from_game': source_game,
            })

    # Analyze each algorithm
    for algorithm in sorted(by_algo_game.keys()):
        print(f"\n{'='*80}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"{'='*80}")

        games_data = by_algo_game[algorithm]

        print(f"\n{'Game':<20} {'Baseline':<15} {'Transfer Avg':<15} {'Transfer Best':<15} {'Benefit':<15}")
        print(f"{'-'*80}")

        for game in sorted(games_data.keys()):
            data = games_data[game]

            # Compute baseline (from-scratch)
            if data['baseline']:
                baseline_rewards = [b['reward'] for b in data['baseline']]
                baseline_avg = sum(baseline_rewards) / len(baseline_rewards)
            else:
                baseline_avg = None

            # Compute transfer stats
            if data['transfers']:
                transfer_rewards = [t['reward'] for t in data['transfers']]
                transfer_avg = sum(transfer_rewards) / len(transfer_rewards)
                transfer_best = max(transfer_rewards)
            else:
                transfer_avg = None
                transfer_best = None

            # Compute benefit
            if baseline_avg is not None and transfer_avg is not None:
                benefit = transfer_avg - baseline_avg
                benefit_pct = (benefit / abs(baseline_avg) * 100) if baseline_avg != 0 else 0
                benefit_str = f"{benefit:+.2f} ({benefit_pct:+.1f}%)"
            else:
                benefit_str = "N/A"

            baseline_str = f"{baseline_avg:.2f}" if baseline_avg is not None else "N/A"
            transfer_avg_str = f"{transfer_avg:.2f}" if transfer_avg is not None else "N/A"
            transfer_best_str = f"{transfer_best:.2f}" if transfer_best is not None else "N/A"

            print(f"{game:<20} {baseline_str:<15} {transfer_avg_str:<15} {transfer_best_str:<15} {benefit_str:<15}")

        # Print detailed transfer breakdown for each game
        print(f"\n{'Game':<20} {'From Source':<20} {'Reward':<15} {'vs Baseline':<15}")
        print(f"{'-'*80}")

        for game in sorted(games_data.keys()):
            data = games_data[game]

            if not data['transfers']:
                continue

            # Get baseline for comparison
            if data['baseline']:
                baseline_rewards = [b['reward'] for b in data['baseline']]
                baseline_avg = sum(baseline_rewards) / len(baseline_rewards)
            else:
                baseline_avg = None

            # Print each transfer result
            for transfer in sorted(data['transfers'], key=lambda x: x['reward'], reverse=True):
                if baseline_avg is not None:
                    diff = transfer['reward'] - baseline_avg
                    diff_str = f"{diff:+.2f}"
                else:
                    diff_str = "N/A"

                print(f"{game:<20} {transfer['from_game']:<20} {transfer['reward']:<15.2f} {diff_str:<15}")

            print()  # Blank line between games


def compare_transfer_strategies(experiments):
    """Compare different transfer learning strategies (freeze, reinit, etc.)."""

    print("\n" + "="*80)
    print("TRANSFER STRATEGY COMPARISON")
    print("="*80)

    # Group by game pair and algorithm
    by_pair_algo = defaultdict(lambda: defaultdict(list))

    for exp in experiments:
        config = exp['config']
        algorithm = config.get('algorithm', 'unknown')
        source_game = config.get('source_game', 'unknown')
        target_game = config.get('target_game', 'unknown')
        pair = f"{source_game}->{target_game}"

        target_stats = compute_summary_stats(exp['target_metrics'])
        if target_stats:
            strategy = []
            if config.get('freeze_encoder', False):
                strategy.append('freeze')
            if config.get('reinit_head', False):
                strategy.append('reinit')
            strategy_str = '+'.join(strategy) if strategy else 'baseline'

            by_pair_algo[pair][algorithm].append({
                'strategy': strategy_str,
                'final_reward': target_stats['final_reward'],
                'max_reward': target_stats['max_reward'],
            })

    # Print comparison for each game pair
    for pair in sorted(by_pair_algo.keys()):
        print(f"\n{'-'*80}")
        print(f"Game Pair: {pair}")
        print(f"{'-'*80}")

        for algorithm in sorted(by_pair_algo[pair].keys()):
            results = by_pair_algo[pair][algorithm]
            print(f"\n  {algorithm.upper()}:")
            for result in results:
                print(f"    {result['strategy']:<20} Final: {result['final_reward']:>8.2f}  Max: {result['max_reward']:>8.2f}")


def generate_summary_report(experiments, output_file):
    """Generate a CSV summary report of all experiments."""

    if not experiments:
        print("No experiments to summarize.")
        return

    print(f"\nGenerating summary report: {output_file}")

    with open(output_file, 'w', newline='') as f:
        fieldnames = [
            'experiment_name',
            'algorithm',
            'source_game',
            'target_game',
            'freeze_encoder',
            'reinit_head',
            'source_timesteps',
            'target_timesteps',
            'source_final_reward',
            'source_max_reward',
            'target_final_reward',
            'target_max_reward',
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for exp in experiments:
            config = exp['config']
            source_stats = compute_summary_stats(exp['source_metrics'])
            target_stats = compute_summary_stats(exp['target_metrics'])

            if source_stats and target_stats:
                writer.writerow({
                    'experiment_name': exp['name'],
                    'algorithm': config.get('algorithm', 'unknown'),
                    'source_game': config.get('source_game', 'unknown'),
                    'target_game': config.get('target_game', 'unknown'),
                    'freeze_encoder': config.get('freeze_encoder', False),
                    'reinit_head': config.get('reinit_head', False),
                    'source_timesteps': source_stats['total_timesteps'],
                    'target_timesteps': target_stats['total_timesteps'],
                    'source_final_reward': source_stats['final_reward'],
                    'source_max_reward': source_stats['max_reward'],
                    'target_final_reward': target_stats['final_reward'],
                    'target_max_reward': target_stats['max_reward'],
                })

    print(f"Summary report saved to: {output_file}")


def print_experiment_status(experiments):
    """Print the status of all experiments."""

    print("\n" + "="*80)
    print("EXPERIMENT STATUS OVERVIEW")
    print("="*80)

    total = len(experiments)
    with_source = sum(1 for exp in experiments if exp['source_metrics'])
    with_target = sum(1 for exp in experiments if exp['target_metrics'])
    complete = sum(1 for exp in experiments if exp['source_metrics'] and exp['target_metrics'])

    print(f"\nTotal experiments found: {total}")
    print(f"With source metrics: {with_source} ({100*with_source/total:.1f}%)")
    print(f"With target metrics: {with_target} ({100*with_target/total:.1f}%)")
    print(f"Fully complete: {complete} ({100*complete/total:.1f}%)")

    # List incomplete experiments
    incomplete = [exp for exp in experiments if not (exp['source_metrics'] and exp['target_metrics'])]
    if incomplete:
        print(f"\nIncomplete experiments ({len(incomplete)}):")
        for exp in incomplete:
            status = []
            if not exp['source_metrics']:
                status.append("no source metrics")
            if not exp['target_metrics']:
                status.append("no target metrics")
            print(f"  - {exp['name']}: {', '.join(status)}")


def main():
    parser = argparse.ArgumentParser(description="Analyze transfer learning experiment results")
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Directory containing experiment results')
    parser.add_argument('--output', type=str, default='results_summary.csv',
                        help='Output CSV file for summary report')
    parser.add_argument('--status-only', action='store_true',
                        help='Only print experiment status overview')

    args = parser.parse_args()

    # Load all experiments
    print(f"Loading experiments from: {args.results_dir}")
    experiments = load_experiment_results(args.results_dir)

    if not experiments:
        print("No experiments found.")
        return

    # Print status
    print_experiment_status(experiments)

    if not args.status_only:
        # Analyze transfer performance
        analyze_transfer_performance(experiments)

        # Analyze transfer learning benefit vs baseline
        analyze_transfer_benefit(experiments)

        # Compare transfer strategies
        compare_transfer_strategies(experiments)

        # Generate summary report
        generate_summary_report(experiments, args.output)

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nDetailed results saved to: {args.output}")
        print("You can open this file in Excel, Google Sheets, or analyze with pandas.")


if __name__ == "__main__":
    main()

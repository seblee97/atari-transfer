#!/usr/bin/env python3
import argparse
import os
import json
from pathlib import Path
from datetime import datetime

def run_transfer_learning(
    algorithm,
    source_game,
    target_game,
    source_timesteps,
    target_timesteps,
    output_dir,
    checkpoint_freq=50000,
    eval_freq=10000,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{algorithm}_{source_game}_to_{target_game}_{timestamp}"
    experiment_dir = os.path.join(output_dir, experiment_name)

    os.makedirs(experiment_dir, exist_ok=True)

    source_checkpoint_dir = os.path.join(experiment_dir, "source_checkpoints")
    source_log_dir = os.path.join(experiment_dir, "source_logs")
    target_checkpoint_dir = os.path.join(experiment_dir, "target_checkpoints")
    target_log_dir = os.path.join(experiment_dir, "target_logs")

    config = {
        "algorithm": algorithm,
        "source_game": source_game,
        "target_game": target_game,
        "source_timesteps": source_timesteps,
        "target_timesteps": target_timesteps,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "experiment_name": experiment_name,
    }

    config_path = os.path.join(experiment_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Starting transfer learning experiment: {experiment_name}")
    print(f"Configuration saved to: {config_path}")

    train_script = f"train_{algorithm}.py"

    print(f"\n{'='*60}")
    print(f"Step 1/2: Training on source game ({source_game})")
    print(f"{'='*60}\n")

    source_cmd = (
        f"python {train_script} "
        f"--game {source_game} "
        f"--timesteps {source_timesteps} "
        f"--checkpoint-dir {source_checkpoint_dir} "
        f"--log-dir {source_log_dir} "
        f"--checkpoint-freq {checkpoint_freq} "
        f"--eval-freq {eval_freq}"
    )

    print(f"Running: {source_cmd}")
    source_result = os.system(source_cmd)

    if source_result != 0:
        print(f"Error: Source game training failed with exit code {source_result}")
        return

    source_model_path = os.path.join(source_checkpoint_dir, "final_model.zip")

    if not os.path.exists(source_model_path):
        print(f"Error: Source model not found at {source_model_path}")
        return

    print(f"\n{'='*60}")
    print(f"Step 2/2: Transfer learning to target game ({target_game})")
    print(f"{'='*60}\n")

    target_cmd = (
        f"python {train_script} "
        f"--game {target_game} "
        f"--timesteps {target_timesteps} "
        f"--checkpoint-dir {target_checkpoint_dir} "
        f"--log-dir {target_log_dir} "
        f"--pretrained {source_model_path} "
        f"--checkpoint-freq {checkpoint_freq} "
        f"--eval-freq {eval_freq}"
    )

    print(f"Running: {target_cmd}")
    target_result = os.system(target_cmd)

    if target_result != 0:
        print(f"Error: Target game training failed with exit code {target_result}")
        return

    print(f"\n{'='*60}")
    print(f"Transfer learning completed successfully!")
    print(f"Results saved to: {experiment_dir}")
    print(f"{'='*60}\n")

    results = {
        "experiment_name": experiment_name,
        "source_model": source_model_path,
        "target_model": os.path.join(target_checkpoint_dir, "final_model.zip"),
        "source_logs": source_log_dir,
        "target_logs": target_log_dir,
    }

    results_path = os.path.join(experiment_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results summary saved to: {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transfer learning between two Atari games")
    parser.add_argument("--algorithm", type=str, required=True, choices=["dqn", "ppo"],
                        help="RL algorithm to use")
    parser.add_argument("--source-game", type=str, required=True,
                        help="Source game name (e.g., Pong, Breakout)")
    parser.add_argument("--target-game", type=str, required=True,
                        help="Target game name (e.g., Pong, Breakout)")
    parser.add_argument("--source-timesteps", type=int, default=1000000,
                        help="Training timesteps for source game")
    parser.add_argument("--target-timesteps", type=int, default=1000000,
                        help="Training timesteps for target game")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Output directory for experiments")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                        help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Evaluation frequency")

    args = parser.parse_args()

    run_transfer_learning(
        algorithm=args.algorithm,
        source_game=args.source_game,
        target_game=args.target_game,
        source_timesteps=args.source_timesteps,
        target_timesteps=args.target_timesteps,
        output_dir=args.output_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
    )

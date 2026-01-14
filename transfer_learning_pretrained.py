#!/usr/bin/env python3
"""
Transfer learning with pre-trained source models.

This script uses pre-trained models from RL Baselines3 Zoo as source models,
then trains on a target game. This saves time by skipping source training.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def run_transfer_with_pretrained(
    algorithm,
    source_game,
    target_game,
    pretrained_model_path,
    target_timesteps=1000000,
    freeze_encoder=False,
    reinit_head=False,
    output_dir="results",
):
    """Run transfer learning using a pre-trained source model."""

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{algorithm}_{source_game}_to_{target_game}_pretrained_{timestamp}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    target_checkpoint_dir = os.path.join(exp_dir, "target_checkpoints")
    target_log_dir = os.path.join(exp_dir, "target_logs")
    os.makedirs(target_checkpoint_dir, exist_ok=True)
    os.makedirs(target_log_dir, exist_ok=True)

    # Save config
    config = {
        "algorithm": algorithm,
        "source_game": source_game,
        "target_game": target_game,
        "pretrained_model": pretrained_model_path,
        "target_timesteps": target_timesteps,
        "freeze_encoder": freeze_encoder,
        "reinit_head": reinit_head,
        "timestamp": timestamp,
    }

    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("=" * 80)
    print("TRANSFER LEARNING WITH PRE-TRAINED MODEL")
    print("=" * 80)
    print(f"Algorithm: {algorithm}")
    print(f"Source game: {source_game} (using pre-trained model)")
    print(f"Target game: {target_game}")
    print(f"Pre-trained model: {pretrained_model_path}")
    print(f"Target timesteps: {target_timesteps}")
    print(f"Freeze encoder: {freeze_encoder}")
    print(f"Reinitialize head: {reinit_head}")
    print(f"Experiment directory: {exp_dir}")
    print("=" * 80)

    # Determine which training script to use
    train_script = f"train_{algorithm}.py"

    if not os.path.exists(train_script):
        print(f"Error: Training script not found: {train_script}")
        return

    # Build target training command
    cmd = [
        "python", train_script,
        "--game", target_game,
        "--timesteps", str(target_timesteps),
        "--checkpoint-dir", target_checkpoint_dir,
        "--log-dir", target_log_dir,
        "--pretrained", pretrained_model_path,
    ]

    if freeze_encoder:
        cmd.append("--freeze-encoder")
    if reinit_head:
        cmd.append("--reinit-head")

    print("\n" + "=" * 80)
    print(f"TRAINING ON TARGET GAME: {target_game}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run target training
    import subprocess
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"Error: Target training failed with code {result.returncode}")
        return

    # Save results metadata
    results = {
        "source_model": pretrained_model_path,
        "target_model": os.path.join(target_checkpoint_dir, "final_model.zip"),
        "target_logs": target_log_dir,
        "config": config_path,
    }

    results_path = os.path.join(exp_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("TRANSFER LEARNING COMPLETE")
    print("=" * 80)
    print(f"Experiment directory: {exp_dir}")
    print(f"Results: {results_path}")
    print(f"Target model: {results['target_model']}")


def main():
    parser = argparse.ArgumentParser(
        description="Transfer learning with pre-trained source models"
    )
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["dqn", "ppo", "qrdqn", "sac"],
                       help="RL algorithm")
    parser.add_argument("--source-game", type=str, required=True,
                       help="Source game (for naming/tracking only)")
    parser.add_argument("--target-game", type=str, required=True,
                       help="Target game to train on")
    parser.add_argument("--pretrained-model", type=str, required=True,
                       help="Path to pre-trained source model (.zip file)")
    parser.add_argument("--target-timesteps", type=int, default=1000000,
                       help="Training timesteps for target game")
    parser.add_argument("--freeze-encoder", action="store_true",
                       help="Freeze CNN encoder during target training")
    parser.add_argument("--reinit-head", action="store_true",
                       help="Reinitialize head layers before target training")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for experiments")

    args = parser.parse_args()

    # Validate pre-trained model exists
    if not os.path.exists(args.pretrained_model):
        print(f"Error: Pre-trained model not found: {args.pretrained_model}")
        print("\nTo download pre-trained models, run:")
        print(f"  python download_pretrained_models.py --algorithm {args.algorithm} --game {args.source_game}")
        return

    run_transfer_with_pretrained(
        algorithm=args.algorithm,
        source_game=args.source_game,
        target_game=args.target_game,
        pretrained_model_path=args.pretrained_model,
        target_timesteps=args.target_timesteps,
        freeze_encoder=args.freeze_encoder,
        reinit_head=args.reinit_head,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

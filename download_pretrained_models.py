#!/usr/bin/env python3
"""
Download pre-trained models from RL Baselines3 Zoo for use as source models.

This script downloads pre-trained DQN, PPO, and other models for Atari games,
which can be used as source models in transfer learning experiments.
"""

import os
import argparse
import requests
from pathlib import Path


# RL Baselines3 Zoo model repository
ZOO_BASE_URL = "https://huggingface.co/sb3"

# Available pre-trained models
# Format: {algorithm: {game: model_name}}
AVAILABLE_MODELS = {
    "dqn": {
        "Pong": "dqn-PongNoFrameskip-v4",
        "Breakout": "dqn-BreakoutNoFrameskip-v4",
        "SpaceInvaders": "dqn-SpaceInvadersNoFrameskip-v4",
        "Qbert": "dqn-QbertNoFrameskip-v4",
        "MsPacman": "dqn-MsPacmanNoFrameskip-v4",
        "Seaquest": "dqn-SeaquestNoFrameskip-v4",
        "BeamRider": "dqn-BeamRiderNoFrameskip-v4",
    },
    "ppo": {
        "Pong": "ppo-PongNoFrameskip-v4",
        "Breakout": "ppo-BreakoutNoFrameskip-v4",
        "SpaceInvaders": "ppo-SpaceInvadersNoFrameskip-v4",
        "Qbert": "ppo-QbertNoFrameskip-v4",
        "MsPacman": "ppo-MsPacmanNoFrameskip-v4",
        "Seaquest": "ppo-SeaquestNoFrameskip-v4",
        "BeamRider": "ppo-BeamRiderNoFrameskip-v4",
    },
    "qrdqn": {
        "Pong": "qrdqn-PongNoFrameskip-v4",
        "Breakout": "qrdqn-BreakoutNoFrameskip-v4",
        "SpaceInvaders": "qrdqn-SpaceInvadersNoFrameskip-v4",
    },
}


def download_model(algorithm, game, output_dir="pretrained_models"):
    """Download a pre-trained model from RL Baselines3 Zoo."""

    if algorithm not in AVAILABLE_MODELS:
        print(f"Error: Algorithm '{algorithm}' not available")
        print(f"Available algorithms: {list(AVAILABLE_MODELS.keys())}")
        return None

    if game not in AVAILABLE_MODELS[algorithm]:
        print(f"Error: Game '{game}' not available for {algorithm}")
        print(f"Available games for {algorithm}: {list(AVAILABLE_MODELS[algorithm].keys())}")
        return None

    model_name = AVAILABLE_MODELS[algorithm][game]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    algo_dir = os.path.join(output_dir, algorithm)
    os.makedirs(algo_dir, exist_ok=True)

    # Construct download URL
    model_url = f"{ZOO_BASE_URL}/{model_name}/resolve/main/{model_name}.zip"
    output_path = os.path.join(algo_dir, f"{game}.zip")

    # Check if already downloaded
    if os.path.exists(output_path):
        print(f"Model already exists: {output_path}")
        return output_path

    print(f"Downloading {algorithm.upper()} model for {game}...")
    print(f"URL: {model_url}")

    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()

        # Download with progress
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)

        print(f"\nSaved to: {output_path}")
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading model: {e}")
        print(f"\nNote: The model might not be available in the RL Baselines3 Zoo.")
        print(f"You may need to train it yourself or check the zoo for available models:")
        print(f"https://huggingface.co/sb3")
        return None


def list_available_models():
    """List all available pre-trained models."""
    print("Available Pre-trained Models:")
    print("=" * 80)

    for algorithm in sorted(AVAILABLE_MODELS.keys()):
        print(f"\n{algorithm.upper()}:")
        games = sorted(AVAILABLE_MODELS[algorithm].keys())
        for game in games:
            print(f"  - {game}")


def download_all_for_experiments(games, algorithms, output_dir="pretrained_models"):
    """Download all models needed for a set of transfer learning experiments."""
    print(f"Downloading models for {len(games)} games × {len(algorithms)} algorithms")
    print("=" * 80)

    success_count = 0
    failed = []

    for algorithm in algorithms:
        for game in games:
            result = download_model(algorithm, game, output_dir)
            if result:
                success_count += 1
            else:
                failed.append((algorithm, game))
            print()  # Blank line between downloads

    print("=" * 80)
    print(f"Successfully downloaded: {success_count} models")

    if failed:
        print(f"Failed to download: {len(failed)} models")
        for algo, game in failed:
            print(f"  - {algo} / {game}")
        print("\nNote: You'll need to train these source models yourself.")


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained Atari models from RL Baselines3 Zoo"
    )
    parser.add_argument("--algorithm", type=str,
                       choices=["dqn", "ppo", "qrdqn"],
                       help="Algorithm to download")
    parser.add_argument("--game", type=str,
                       help="Game to download (e.g., Pong, Breakout)")
    parser.add_argument("--games", type=str, nargs="+",
                       help="Multiple games to download")
    parser.add_argument("--algorithms", type=str, nargs="+",
                       choices=["dqn", "ppo", "qrdqn"],
                       help="Multiple algorithms to download")
    parser.add_argument("--output-dir", type=str, default="pretrained_models",
                       help="Output directory for downloaded models")
    parser.add_argument("--list", action="store_true",
                       help="List all available pre-trained models")
    parser.add_argument("--all", action="store_true",
                       help="Download all available models")

    args = parser.parse_args()

    if args.list:
        list_available_models()
        return

    if args.all:
        print("Downloading ALL available pre-trained models...")
        for algorithm in AVAILABLE_MODELS.keys():
            for game in AVAILABLE_MODELS[algorithm].keys():
                download_model(algorithm, game, args.output_dir)
                print()
        return

    # Download specific models
    if args.algorithm and args.game:
        download_model(args.algorithm, args.game, args.output_dir)

    # Download for multiple games and algorithms
    elif args.games and args.algorithms:
        download_all_for_experiments(args.games, args.algorithms, args.output_dir)

    else:
        print("Error: Must specify --algorithm and --game, or --games and --algorithms")
        print("Use --list to see available models")
        print("Use --help for more options")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Phased training script that can train in chunks and resume from checkpoints.

This script supports:
1. Training from scratch for N timesteps
2. Resuming from a checkpoint and training for additional timesteps
3. Creating checkpoint files that signal completion for dependency chaining
"""
import argparse
import os
from pathlib import Path

import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import DQN, PPO, SAC
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

# Register ALE environments
gym.register_envs(ale_py)

def make_atari_env(game_name):
    def _init():
        env = gym.make(f"ALE/{game_name}-v5", render_mode=None)
        env = AtariWrapper(env)
        return env
    return _init

def get_algorithm_class(algorithm):
    """Get the algorithm class from string name."""
    algorithms = {
        "dqn": DQN,
        "ppo": PPO,
        "qrdqn": QRDQN,
        "sac": SAC,
    }
    if algorithm.lower() not in algorithms:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(algorithms.keys())}")
    return algorithms[algorithm.lower()]

def get_algorithm_hyperparams(algorithm):
    """Get default hyperparameters for each algorithm."""
    if algorithm.lower() == "dqn":
        return {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 100000,
            "batch_size": 32,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
        }
    elif algorithm.lower() == "qrdqn":
        return {
            "learning_rate": 1e-4,
            "buffer_size": 100000,
            "learning_starts": 100000,
            "batch_size": 32,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_final_eps": 0.01,
        }
    elif algorithm.lower() == "ppo":
        return {
            "learning_rate": 2.5e-4,
            "n_steps": 128,
            "batch_size": 256,
            "n_epochs": 4,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
        }
    elif algorithm.lower() == "sac":
        return {
            "learning_rate": 3e-4,
            "buffer_size": 100000,
            "learning_starts": 100,
            "batch_size": 64,
            "tau": 0.005,
            "gamma": 0.99,
        }
    else:
        return {}

def train_phased(
    algorithm,
    game_name,
    phase_timesteps,
    checkpoint_dir,
    log_dir,
    resume_from=None,
    checkpoint_freq=50000,
    eval_freq=10000,
    phase_id=None,
    signal_file=None,
):
    """
    Train a model for a specific phase (number of timesteps).

    Args:
        algorithm: Algorithm name (dqn, ppo, qrdqn, sac)
        game_name: Atari game name
        phase_timesteps: Number of timesteps to train in this phase
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        resume_from: Path to checkpoint to resume from (None for training from scratch)
        checkpoint_freq: Frequency of checkpoint saves
        eval_freq: Frequency of evaluations
        phase_id: Identifier for this phase (e.g., "phase_1")
        signal_file: Path to signal file to create when phase completes
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Create environment
    env = DummyVecEnv([make_atari_env(game_name)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_atari_env(game_name)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # Configure logger
    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix=f"{algorithm}_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    # Get algorithm class and hyperparameters
    AlgorithmClass = get_algorithm_class(algorithm)
    hyperparams = get_algorithm_hyperparams(algorithm)

    # Load or create model
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")

        # Override incompatible buffer settings for loading
        custom_objects = {
            "optimize_memory_usage": False,
            "handle_timeout_termination": False,
        }

        model = AlgorithmClass.load(resume_from, env=env, custom_objects=custom_objects)
        model.set_logger(logger)

        print(f"Successfully loaded checkpoint. Continuing training for {phase_timesteps} timesteps.")
    else:
        print(f"Training {algorithm.upper()} from scratch on {game_name}")

        model = AlgorithmClass(
            "CnnPolicy",
            env,
            **hyperparams,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.set_logger(logger)

    # Train for this phase
    print(f"Training for {phase_timesteps} timesteps (Phase: {phase_id or 'N/A'})")
    model.learn(
        total_timesteps=phase_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=False if resume_from else True,
    )

    # Save the final checkpoint for this phase
    phase_checkpoint = os.path.join(checkpoint_dir, f"{phase_id or 'final'}_model.zip")
    model.save(phase_checkpoint)
    print(f"Phase checkpoint saved to {phase_checkpoint}")

    # Create signal file if requested (for dependency chaining)
    if signal_file:
        signal_path = Path(signal_file)
        signal_path.parent.mkdir(parents=True, exist_ok=True)
        signal_path.touch()
        print(f"Signal file created: {signal_file}")

    env.close()
    eval_env.close()

    return phase_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phased training for Atari RL")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["dqn", "ppo", "qrdqn", "sac"],
                       help="RL algorithm")
    parser.add_argument("--game", type=str, required=True,
                       help="Atari game name (e.g., Pong, Breakout)")
    parser.add_argument("--phase-timesteps", type=int, required=True,
                       help="Number of timesteps to train in this phase")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory for logs")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint-freq", type=int, default=50000,
                       help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")
    parser.add_argument("--phase-id", type=str, default=None,
                       help="Identifier for this phase")
    parser.add_argument("--signal-file", type=str, default=None,
                       help="Path to signal file to create when phase completes")

    args = parser.parse_args()

    train_phased(
        algorithm=args.algorithm,
        game_name=args.game,
        phase_timesteps=args.phase_timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        resume_from=args.resume_from,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        phase_id=args.phase_id,
        signal_file=args.signal_file,
    )

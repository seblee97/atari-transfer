#!/usr/bin/env python3
"""
Continuous training script that saves checkpoints at specified intervals.

This script trains a model continuously from 0 to N timesteps, saving checkpoints
at regular intervals. Unlike train_phased.py, this never stops and resumes - it's
one continuous training run.

The checkpoints can be used by transfer learning jobs that start as soon as each
checkpoint is created.
"""
import argparse
import os
from pathlib import Path

import gymnasium as gym
import ale_py
from stable_baselines3 import DQN, PPO, SAC
from sb3_contrib import QRDQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
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


class NamedCheckpointCallback(BaseCallback):
    """
    Callback for saving checkpoints at specific intervals with custom names.

    This is like CheckpointCallback but saves with predictable names based on
    the number of timesteps, making it easy for transfer jobs to know when
    checkpoints are available.
    """
    def __init__(self, save_freq, save_path, name_prefix="model",
                 checkpoint_intervals=None, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.checkpoint_intervals = checkpoint_intervals or []
        self.saved_checkpoints = set()

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if we should save based on checkpoint intervals
        current_timesteps = self.num_timesteps

        for interval in self.checkpoint_intervals:
            if current_timesteps >= interval and interval not in self.saved_checkpoints:
                checkpoint_path = os.path.join(
                    self.save_path,
                    f"{self.name_prefix}_{interval}.zip"
                )
                self.model.save(checkpoint_path)
                self.saved_checkpoints.add(interval)
                if self.verbose > 0:
                    print(f"Saved checkpoint at {current_timesteps:,} steps: {checkpoint_path}")

        return True


def train_continuous_with_checkpoints(
    algorithm,
    game_name,
    total_timesteps,
    checkpoint_dir,
    log_dir,
    checkpoint_intervals=None,
    eval_freq=10000,
):
    """
    Train a model continuously with periodic checkpoints.

    Args:
        algorithm: Algorithm name (dqn, ppo, qrdqn, sac)
        game_name: Atari game name
        total_timesteps: Total timesteps to train
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory to save logs
        checkpoint_intervals: List of timestep counts where checkpoints should be saved
                             e.g., [10000000, 20000000, 30000000] for 10M, 20M, 30M
        eval_freq: Frequency of evaluations
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

    # Get algorithm class and hyperparameters
    AlgorithmClass = get_algorithm_class(algorithm)
    hyperparams = get_algorithm_hyperparams(algorithm)

    # Create model
    print(f"Training {algorithm.upper()} on {game_name} for {total_timesteps:,} timesteps")
    print(f"Checkpoints will be saved at: {checkpoint_intervals}")

    model = AlgorithmClass(
        "CnnPolicy",
        env,
        **hyperparams,
        verbose=1,
        tensorboard_log=log_dir,
    )
    model.set_logger(logger)

    # Setup callbacks
    checkpoint_callback = NamedCheckpointCallback(
        save_freq=eval_freq,  # Check every eval
        save_path=checkpoint_dir,
        name_prefix="checkpoint",
        checkpoint_intervals=checkpoint_intervals,
        verbose=1,
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

    # Train continuously
    print(f"Starting continuous training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    eval_env.close()

    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continuous training with checkpoints")
    parser.add_argument("--algorithm", type=str, required=True,
                       choices=["dqn", "ppo", "qrdqn", "sac"],
                       help="RL algorithm")
    parser.add_argument("--game", type=str, required=True,
                       help="Atari game name")
    parser.add_argument("--timesteps", type=int, required=True,
                       help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                       help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, required=True,
                       help="Directory for logs")
    parser.add_argument("--checkpoint-intervals", type=int, nargs="+", required=True,
                       help="Timesteps at which to save checkpoints (e.g., 10000000 20000000 30000000)")
    parser.add_argument("--eval-freq", type=int, default=10000,
                       help="Evaluation frequency")

    args = parser.parse_args()

    train_continuous_with_checkpoints(
        algorithm=args.algorithm,
        game_name=args.game,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        checkpoint_intervals=args.checkpoint_intervals,
        eval_freq=args.eval_freq,
    )

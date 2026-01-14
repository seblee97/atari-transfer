#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure

def make_atari_env(game_name):
    def _init():
        env = gym.make(f"ALE/{game_name}-v5", render_mode=None)
        env = AtariWrapper(env)
        return env
    return _init

def train_dqn(
    game_name,
    total_timesteps,
    checkpoint_dir,
    log_dir,
    pretrained_model=None,
    checkpoint_freq=50000,
    eval_freq=10000,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=100000,
    batch_size=32,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_atari_env(game_name)])
    env = VecFrameStack(env, n_stack=4)

    eval_env = DummyVecEnv([make_atari_env(game_name)])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    logger = configure(log_dir, ["stdout", "csv", "tensorboard"])

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="dqn_model",
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

    if pretrained_model and os.path.exists(pretrained_model):
        print(f"Loading pretrained model from {pretrained_model}")
        model = DQN.load(pretrained_model, env=env)
        model.set_logger(logger)
    else:
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_final_eps=exploration_final_eps,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.set_logger(logger)

    print(f"Training DQN on {game_name} for {total_timesteps} timesteps")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=10,
        progress_bar=True,
    )

    final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    env.close()
    eval_env.close()

    return final_model_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on Atari game")
    parser.add_argument("--game", type=str, required=True, help="Atari game name (e.g., Pong, Breakout)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")

    args = parser.parse_args()

    train_dqn(
        game_name=args.game,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        pretrained_model=args.pretrained,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
    )

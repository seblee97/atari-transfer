#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import gymnasium as gym
import ale_py
import torch
from stable_baselines3 import SAC
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

# Note: SAC in stable-baselines3 is designed for continuous action spaces.
# For discrete action Atari games, consider using DQN, PPO, or QR-DQN instead.
# This implementation may not work optimally with discrete action spaces.

def train_sac(
    game_name,
    total_timesteps,
    checkpoint_dir,
    log_dir,
    pretrained_model=None,
    checkpoint_freq=50000,
    eval_freq=10000,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=100000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    freeze_encoder=False,
    reinit_head=False,
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
        name_prefix="sac_model",
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
        model = SAC.load(pretrained_model, env=env)
        model.set_logger(logger)

        # Handle freeze encoder and reinit head
        if freeze_encoder or reinit_head:
            policy = model.policy

            if reinit_head:
                print("Reinitializing actor and critic heads")
                # SAC has actor and critic networks
                if hasattr(policy, 'actor') and hasattr(policy.actor, 'mu'):
                    torch.nn.init.orthogonal_(policy.actor.mu.weight, gain=0.01)
                    torch.nn.init.constant_(policy.actor.mu.bias, 0.0)
                    if hasattr(policy.actor, 'log_std'):
                        torch.nn.init.orthogonal_(policy.actor.log_std.weight, gain=0.01)
                        torch.nn.init.constant_(policy.actor.log_std.bias, 0.0)

                # Reinitialize critic networks
                for critic in [model.critic, model.critic_target]:
                    if hasattr(critic, 'q_networks'):
                        for q_net in critic.q_networks:
                            final_layer = list(q_net.children())[-1]
                            if isinstance(final_layer, torch.nn.Linear):
                                torch.nn.init.orthogonal_(final_layer.weight, gain=1)
                                torch.nn.init.constant_(final_layer.bias, 0.0)

            if freeze_encoder:
                print("Freezing CNN encoder layers")
                # Freeze the shared feature extractor (CNN)
                if hasattr(policy, 'features_extractor'):
                    for param in policy.features_extractor.parameters():
                        param.requires_grad = False

                # Freeze critic feature extractors
                if hasattr(model.critic, 'features_extractor'):
                    for param in model.critic.features_extractor.parameters():
                        param.requires_grad = False
                if hasattr(model.critic_target, 'features_extractor'):
                    for param in model.critic_target.features_extractor.parameters():
                        param.requires_grad = False

                print(f"Trainable parameters: {sum(p.numel() for p in model.policy.parameters() if p.requires_grad)}")
                print(f"Frozen parameters: {sum(p.numel() for p in model.policy.parameters() if not p.requires_grad)}")
    else:
        model = SAC(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            verbose=1,
            tensorboard_log=log_dir,
        )
        model.set_logger(logger)

    print(f"Training SAC on {game_name} for {total_timesteps} timesteps")
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
    parser = argparse.ArgumentParser(description="Train SAC on Atari game")
    parser.add_argument("--game", type=str, required=True, help="Atari game name (e.g., Pong, Breakout)")
    parser.add_argument("--timesteps", type=int, default=1000000, help="Total training timesteps")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for logs")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to pretrained model")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Checkpoint frequency")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Evaluation frequency")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--tau", type=float, default=0.005, help="Target network update rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--freeze-encoder", action="store_true", help="Freeze CNN encoder layers during transfer")
    parser.add_argument("--reinit-head", action="store_true", help="Reinitialize the actor and critic head weights")

    args = parser.parse_args()

    train_sac(
        game_name=args.game,
        total_timesteps=args.timesteps,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        pretrained_model=args.pretrained,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        tau=args.tau,
        gamma=args.gamma,
        freeze_encoder=args.freeze_encoder,
        reinit_head=args.reinit_head,
    )

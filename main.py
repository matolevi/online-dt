""""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import argparse
import pickle
import random
import time
import time
import os
try:
    import gym
except ImportError:
    import gymnasium as gym
import torch
import numpy as np
import just_d4rl as d4rl

# Register D4RL environments with gym using just_d4rl
import sys
sys.modules['d4rl'] = d4rl
import d4rl_envs  # Register D4RL environments with gymnasium

import utils
from replay_buffer import ReplayBuffer
from lamb import Lamb
from stable_baselines3.common.vec_env import SubprocVecEnv
from pathlib import Path
from data import create_dataloader
from decision_transformer.models.decision_transformer import DecisionTransformer
from evaluation import create_vec_eval_episodes_fn, vec_evaluate_episode_rtg
from trainer import SequenceTrainer
from logger import Logger

# Disable TensorBoard for Apple Silicon compatibility
TENSORBOARD_AVAILABLE = False
SummaryWriter = None

MAX_EPISODE_LEN = 1000


class Experiment:
    def __init__(self, variant):
        print("DEBUG: Initializing Experiment...")
        
        print("DEBUG: Getting environment spec...")
        self.state_dim, self.act_dim, self.action_range = self._get_env_spec(variant)
        print(f"DEBUG: Environment spec - state_dim: {self.state_dim}, act_dim: {self.act_dim}, action_range: {self.action_range}")
        
        print("DEBUG: Loading dataset...")
        self.offline_trajs, self.state_mean, self.state_std = self._load_dataset(
            variant["env"]
        )
        print(f"DEBUG: Dataset loaded - {len(self.offline_trajs)} trajectories")
        
        # initialize by offline trajs
        print("DEBUG: Initializing replay buffer...")
        self.replay_buffer = ReplayBuffer(variant["replay_size"], self.offline_trajs)
        print("DEBUG: Replay buffer initialized")

        self.aug_trajs = []

        self.device = variant.get("device", "cuda")
        print(f"DEBUG: Using device: {self.device}")
        
        self.target_entropy = -self.act_dim
        model_type = variant.get("model_type", "dt")
        print(f"DEBUG: Initializing model type: {model_type}")
        
        if model_type == "dt":
            from decision_transformer.models.decision_transformer import DecisionTransformer as Agent
            print("DEBUG: Creating Decision Transformer model...")
            self.model = Agent(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                action_range=self.action_range,
                max_length=variant["K"],
                eval_context_length=variant["eval_context_length"],
                max_ep_len=MAX_EPISODE_LEN,
                hidden_size=variant["embed_dim"],
                n_layer=variant["n_layer"],
                n_head=variant["n_head"],
                n_inner=4 * variant["embed_dim"],
                activation_function=variant["activation_function"],
                n_positions=1024,
                resid_pdrop=variant["dropout"],
                attn_pdrop=variant["dropout"],
                stochastic_policy=True,
                ordering=variant["ordering"],
                init_temperature=variant["init_temperature"],
                target_entropy=self.target_entropy,
            ).to(device=self.device)
            print("DEBUG: Decision Transformer model created successfully")
        elif model_type in ("dmamba", "dmamba-min"):
            from mamba_models.decision_mamba import DecisionMamba as Agent
            print("DEBUG: Creating Decision Mamba model...")
            self.model = Agent(
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                hidden_size=variant["embed_dim"],
                max_length=variant["K"],
                max_ep_len=MAX_EPISODE_LEN,
                action_tanh=True,
                remove_act_embs=(model_type == "dmamba-min"),
                stochastic_policy=True,
                init_temperature=variant["init_temperature"],
                target_entropy=self.target_entropy,
                action_range=self.action_range,
                n_layer=variant["n_layer"],
                n_head=variant["n_head"],
                n_inner=4 * variant["embed_dim"],
                activation_function=variant["activation_function"],
                n_positions=1024,
                resid_pdrop=variant["dropout"],
                attn_pdrop=variant["dropout"],
            ).to(device=self.device)
            print("DEBUG: Decision Mamba model created successfully")
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        print("DEBUG: Setting up optimizers...")
        self.optimizer = Lamb(
            self.model.parameters(),
            lr=variant["learning_rate"],
            weight_decay=variant["weight_decay"],
            eps=1e-8,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda steps: min((steps + 1) / variant["warmup_steps"], 1)
        )

        # Only create log_temperature_optimizer for stochastic models
        if hasattr(self.model, 'log_temperature'):
            self.log_temperature_optimizer = torch.optim.Adam(
                [self.model.log_temperature],
                lr=1e-4,
                betas=[0.9, 0.999],
            )
        else:
            self.log_temperature_optimizer = None
        print("DEBUG: Optimizers set up successfully")

        # track the training progress and
        # training/evaluation/online performance in all the iterations
        self.pretrain_iter = 0
        self.online_iter = 0
        self.total_transitions_sampled = 0
        self.variant = variant
        self.reward_scale = 1.0 if "antmaze" in variant["env"] else 0.001
        print(f"DEBUG: Using reward scale: {self.reward_scale}")
        
        self.logger = Logger(variant)
        print("DEBUG: Logger initialized")
        print("DEBUG: Experiment initialization completed successfully!")

    def _get_env_spec(self, variant):
        env = gym.make(variant["env"])
        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        action_range = [
            float(env.action_space.low.min()) + 1e-6,
            float(env.action_space.high.max()) - 1e-6,
        ]
        env.close()
        return state_dim, act_dim, action_range

    def _save_model(self, path_prefix, is_pretrain_model=False):
        to_save = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "pretrain_iter": self.pretrain_iter,
            "online_iter": self.online_iter,
            "args": self.variant,
            "total_transitions_sampled": self.total_transitions_sampled,
            "np": np.random.get_state(),
            "python": random.getstate(),
            "pytorch": torch.get_rng_state(),
            "log_temperature_optimizer_state_dict": self.log_temperature_optimizer.state_dict() if self.log_temperature_optimizer else None,
        }

        with open(f"{path_prefix}/model.pt", "wb") as f:
            torch.save(to_save, f)
        print(f"\nModel saved at {path_prefix}/model.pt")

        if is_pretrain_model:
            with open(f"{path_prefix}/pretrain_model.pt", "wb") as f:
                torch.save(to_save, f)
            print(f"Model saved at {path_prefix}/pretrain_model.pt")

    def _load_model(self, path_prefix):
        if Path(f"{path_prefix}/model.pt").exists():
            with open(f"{path_prefix}/model.pt", "rb") as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if self.log_temperature_optimizer is not None and checkpoint["log_temperature_optimizer_state_dict"] is not None:
                self.log_temperature_optimizer.load_state_dict(
                    checkpoint["log_temperature_optimizer_state_dict"]
                )
            self.pretrain_iter = checkpoint["pretrain_iter"]
            self.online_iter = checkpoint["online_iter"]
            self.total_transitions_sampled = checkpoint["total_transitions_sampled"]
            np.random.set_state(checkpoint["np"])
            random.setstate(checkpoint["python"])
            torch.set_rng_state(checkpoint["pytorch"])
            print(f"Model loaded at {path_prefix}/model.pt")

    def _load_dataset(self, env_name):

        dataset_path = f"./data/{env_name}.pkl"
        with open(dataset_path, "rb") as f:
            trajectories = pickle.load(f)

        states, traj_lens, returns = [], [], []
        for path in trajectories:
            states.append(path["observations"])
            traj_lens.append(len(path["observations"]))
            returns.append(path["rewards"].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print("=" * 50)
        print(f"Starting new experiment: {env_name}")
        print(f"{len(traj_lens)} trajectories, {num_timesteps} timesteps found")
        print(f"Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}")
        print(f"Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}")
        print(f"Average length: {np.mean(traj_lens):.2f}, std: {np.std(traj_lens):.2f}")
        print(f"Max length: {np.max(traj_lens):.2f}, min: {np.min(traj_lens):.2f}")
        print("=" * 50)

        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(trajectories) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        trajectories = [trajectories[ii] for ii in sorted_inds]

        return trajectories, state_mean, state_std

    def render_evaluation(self):
        '''Run rendering evaluation'''
        if not self.variant.get("render", False):
            return {}
        
        print("\n" + "="*50)
        print("🎬 RENDERING EVALUATION")
        print("="*50)
        
        # Import the render function
        from evaluation import render_episode
        
        returns = []
        lengths = []
        
        # Run multiple rendering episodes if requested
        for i in range(self.variant.get("render_episodes", 1)):
            print(f"\nRendering episode {i+1}/{self.variant.get('render_episodes', 1)}")
            
            episode_return, episode_length = render_episode(
                env_name=self.variant["env"],
                model=self.model,
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                target_return=self.variant["eval_rtg"] * self.reward_scale,
                max_ep_len=MAX_EPISODE_LEN,
                reward_scale=self.reward_scale,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,  # Use mean action for more stable visualization
                render_mode="human",
                fps=self.variant.get("render_fps", 30),
            )
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        avg_return = np.mean(returns)
        avg_length = np.mean(lengths)
        
        print(f"\nRendering Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print("="*50 + "\n")
        
        return {
            "render/return_mean": avg_return,
            "render/return_std": np.std(returns) if len(returns) > 1 else 0.0,
            "render/length_mean": avg_length,
            "render/length_std": np.std(lengths) if len(lengths) > 1 else 0.0,
        }


    def render_evaluation_video(self):
        '''Run rendering evaluation and save as video'''
        if not self.variant.get("render", False):
            return {}
        
        print("\n" + "="*50)
        print("🎬 VIDEO RECORDING EVALUATION")
        print("="*50)
        
        # Import the video render function
        from evaluation import render_episode_video
        
        returns = []
        lengths = []
        
        # Create video folder path
        video_folder = os.path.join(self.logger.log_path, "videos")
        
        # Run multiple rendering episodes if requested
        for i in range(self.variant.get("render_episodes", 1)):
            print(f"\nRecording episode {i+1}/{self.variant.get('render_episodes', 1)}")
            
            episode_return, episode_length = render_episode_video(
                env_name=self.variant["env"],
                model=self.model,
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                target_return=self.variant["eval_rtg"] * self.reward_scale,
                max_ep_len=MAX_EPISODE_LEN,
                reward_scale=self.reward_scale,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,  # Use mean action for more stable visualization
                video_folder=video_folder,
                fps=self.variant.get("render_fps", 30),
            )
            
            returns.append(episode_return)
            lengths.append(episode_length)
        
        avg_return = np.mean(returns)
        avg_length = np.mean(lengths)
        
        print(f"\nVideo Recording Summary:")
        print(f"  Average Return: {avg_return:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Videos saved in: {video_folder}")
        print("="*50 + "\n")
        
        return {
            "video/return_mean": avg_return,
            "video/return_std": np.std(returns) if len(returns) > 1 else 0.0,
            "video/length_mean": avg_length,
            "video/length_std": np.std(lengths) if len(lengths) > 1 else 0.0,
        }




    def _augment_trajectories(
        self,
        online_envs,
        target_explore,
        n,
        randomized=False,
    ):

        max_ep_len = MAX_EPISODE_LEN

        with torch.no_grad():
            # generate init state
            target_return = [target_explore * self.reward_scale] * online_envs.num_envs

            returns, lengths, trajs = vec_evaluate_episode_rtg(
                online_envs,
                self.state_dim,
                self.act_dim,
                self.model,
                max_ep_len=min(max_ep_len, 20),  # Use very short episodes for online rollouts during testing
                reward_scale=self.reward_scale,
                target_return=target_return,
                mode="normal",
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=False,
                verbose=self.variant.get("verbose", 1),  # Add verbose parameter
            )

        self.replay_buffer.add_new_trajs(trajs)
        self.aug_trajs += trajs
        self.total_transitions_sampled += np.sum(lengths)

        return {
            "aug_traj/return": np.mean(returns),
            "aug_traj/length": np.mean(lengths),
        }

    def pretrain(self, eval_envs, loss_fn):
        print("\n\n\n*** Pretrain ***")
        print(f"DEBUG: Starting pretrain phase for {self.variant['max_pretrain_iters']} iterations")

        print("DEBUG: Creating evaluation functions...")
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                verbose=self.variant.get("verbose", 1),
            )
        ]
        print("DEBUG: Evaluation functions created")

        print("DEBUG: Creating sequence trainer...")
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            verbose=self.variant.get("verbose", 1),
        )
        print("DEBUG: Sequence trainer created")

        writer = (
            SummaryWriter(self.logger.log_path) if (self.variant["log_to_tb"] and TENSORBOARD_AVAILABLE) else None
        )
        print(f"DEBUG: TensorBoard writer: {'enabled' if writer else 'disabled'}")
        
        while self.pretrain_iter < self.variant["max_pretrain_iters"]:
            print(f"\nDEBUG: Starting pretrain iteration {self.pretrain_iter + 1}/{self.variant['max_pretrain_iters']}")
            
            # in every iteration, prepare the data loader
            print("DEBUG: Creating dataloader...")
            dataloader = create_dataloader(
                trajectories=self.offline_trajs,
                num_iters=self.variant["num_updates_per_pretrain_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                verbose=self.variant.get("verbose", 1),
            )
            print(f"DEBUG: Dataloader created with {self.variant['num_updates_per_pretrain_iter']} updates per iteration")

            print("DEBUG: Starting training iteration...")
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            print(f"DEBUG: Training iteration completed. Loss: {train_outputs.get('training/train_loss_mean', 'N/A')}")
            
            print("DEBUG: Starting evaluation...")
            eval_outputs, eval_reward = self.evaluate(eval_fns)
            print(f"DEBUG: Evaluation completed. Reward: {eval_reward}")
            
            if (self.variant.get("render", False) and (self.pretrain_iter + 1) % self.variant.get("render_interval", 3) == 0):
                if self.variant.get("save_video", False):
                    render_outputs = self.render_evaluation_video()
                else:
                    render_outputs = self.render_evaluation()
                outputs.update(render_outputs)


            outputs = {"time/total": time.time() - self.start_time}
            outputs.update(train_outputs)
            outputs.update(eval_outputs)
            
            print("DEBUG: Logging metrics...")
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            print("DEBUG: Saving model...")
            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=True,
            )

            self.pretrain_iter += 1
            print(f"DEBUG: Pretrain iteration {self.pretrain_iter} completed")

        print("DEBUG: Pretrain phase completed!")

    def evaluate(self, eval_fns):
        eval_start = time.time()
        self.model.eval()
        outputs = {}
        for eval_fn in eval_fns:
            o = eval_fn(self.model)
            outputs.update(o)
        outputs["time/evaluation"] = time.time() - eval_start

        eval_reward = outputs["evaluation/return_mean_gm"]
        return outputs, eval_reward

    def online_tuning(self, online_envs, eval_envs, loss_fn):

        print("\n\n\n*** Online Finetuning ***")
        print(f"DEBUG: Starting online tuning for {self.variant['max_online_iters']} iterations")

        print("DEBUG: Creating sequence trainer for online tuning...")
        trainer = SequenceTrainer(
            model=self.model,
            optimizer=self.optimizer,
            log_temperature_optimizer=self.log_temperature_optimizer,
            scheduler=self.scheduler,
            device=self.device,
            verbose=self.variant.get("verbose", 1),
        )
        
        print("DEBUG: Creating evaluation functions for online tuning...")
        eval_fns = [
            create_vec_eval_episodes_fn(
                vec_env=eval_envs,
                eval_rtg=self.variant["eval_rtg"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                device=self.device,
                use_mean=True,
                reward_scale=self.reward_scale,
                verbose=self.variant.get("verbose", 1),
            )
        ]
        
        writer = (
            SummaryWriter(self.logger.log_path) if (self.variant["log_to_tb"] and TENSORBOARD_AVAILABLE) else None
        )
        print(f"DEBUG: TensorBoard writer: {'enabled' if writer else 'disabled'}")
        
        while self.online_iter < self.variant["max_online_iters"]:
            print(f"\nDEBUG: Starting online iteration {self.online_iter + 1}/{self.variant['max_online_iters']}")

            outputs = {}
            
            print("DEBUG: Augmenting trajectories with online rollouts...")
            augment_outputs = self._augment_trajectories(
                online_envs,
                self.variant["online_rtg"],
                n=self.variant["num_online_rollouts"],
            )
            outputs.update(augment_outputs)
            print(f"DEBUG: Trajectory augmentation completed. Return: {augment_outputs.get('aug_traj/return', 'N/A')}")

            print("DEBUG: Creating dataloader for online training...")
            dataloader = create_dataloader(
                trajectories=self.replay_buffer.trajectories,
                num_iters=self.variant["num_updates_per_online_iter"],
                batch_size=self.variant["batch_size"],
                max_len=self.variant["K"],
                state_dim=self.state_dim,
                act_dim=self.act_dim,
                state_mean=self.state_mean,
                state_std=self.state_std,
                reward_scale=self.reward_scale,
                action_range=self.action_range,
                verbose=self.variant.get("verbose", 1),
            )

            # finetuning
            is_last_iter = self.online_iter == self.variant["max_online_iters"] - 1
            if (self.online_iter + 1) % self.variant[
                "eval_interval"
            ] == 0 or is_last_iter:
                evaluation = True
            else:
                evaluation = False

            print(f"DEBUG: Starting training iteration (evaluation: {evaluation})...")
            train_outputs = trainer.train_iteration(
                loss_fn=loss_fn,
                dataloader=dataloader,
            )
            outputs.update(train_outputs)
            print(f"DEBUG: Training iteration completed. Loss: {train_outputs.get('training/train_loss_mean', 'N/A')}")

            if evaluation:
                print("DEBUG: Running evaluation...")
                eval_outputs, eval_reward = self.evaluate(eval_fns)
                outputs.update(eval_outputs)
                print(f"DEBUG: Evaluation completed. Reward: {eval_reward}")
                if self.variant.get("render", False):
                    if self.variant.get("save_video", False):
                        render_outputs = self.render_evaluation_video()
                    else:
                        render_outputs = self.render_evaluation()
                    outputs.update(render_outputs)


            outputs["time/total"] = time.time() - self.start_time

            # log the metrics
            print("DEBUG: Logging metrics...")
            self.logger.log_metrics(
                outputs,
                iter_num=self.pretrain_iter + self.online_iter,
                total_transitions_sampled=self.total_transitions_sampled,
                writer=writer,
            )

            print("DEBUG: Saving model...")
            self._save_model(
                path_prefix=self.logger.log_path,
                is_pretrain_model=False,
            )

            self.online_iter += 1
            print(f"DEBUG: Online iteration {self.online_iter} completed")

        print("DEBUG: Online tuning phase completed!")

    def __call__(self):
        print("DEBUG: Starting experiment execution...")

        utils.set_seed_everywhere(args.seed)
        print(f"DEBUG: Seed set to {args.seed}")

        def loss_fn(
            a_hat_dist,
            a,
            attention_mask,
            entropy_reg,
        ):
            # a_hat is a SquashedNormal Distribution
            log_likelihood = a_hat_dist.log_likelihood(a)[attention_mask > 0].mean()

            entropy = a_hat_dist.entropy().mean()
            loss = -(log_likelihood + entropy_reg * entropy)

            return (
                loss,
                -log_likelihood,
                entropy,
            )

        def get_env_builder(seed, env_name, target_goal=None):
            def make_env_fn():
                env = gym.make(env_name)
                # Handle both old gym and new gymnasium APIs
                if hasattr(env, 'seed'):
                    env.seed(seed)
                    if hasattr(env.env, "wrapped_env"):
                        env.env.wrapped_env.seed(seed)
                    elif hasattr(env.env, "seed"):
                        env.env.seed(seed)
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

                if target_goal:
                    env.set_target_goal(target_goal)
                    print(f"Set the target goal to be {env.target_goal}")
                return env

            return make_env_fn

        print("\n\nDEBUG: Making Eval Env.....")
        env_name = self.variant["env"]
        if "antmaze" in env_name:
            print("DEBUG: Creating antmaze environment to get target goal...")
            env = gym.make(env_name)
            target_goal = env.target_goal
            env.close()
            print(f"Generated the fixed target goal: {target_goal}")
        else:
            target_goal = None
            print("DEBUG: No target goal needed for this environment")
            
        print(f"DEBUG: Creating {self.variant['num_eval_episodes']} evaluation environments...")
        eval_envs = SubprocVecEnv(
            [
                get_env_builder(i, env_name=env_name, target_goal=target_goal)
                for i in range(self.variant["num_eval_episodes"])
            ]
        )
        print("DEBUG: Evaluation environments created successfully")

        self.start_time = time.time()
        print(f"DEBUG: Experiment start time: {self.start_time}")
        
        if self.variant["max_pretrain_iters"]:
            print(f"DEBUG: Starting pretrain phase with {self.variant['max_pretrain_iters']} iterations...")
            self.pretrain(eval_envs, loss_fn)
            print("DEBUG: Pretrain phase completed")
        else:
            print("DEBUG: Skipping pretrain phase (max_pretrain_iters = 0)")

        if self.variant["max_online_iters"]:
            print(f"\n\nDEBUG: Making Online Env for {self.variant['num_online_rollouts']} rollouts.....")
            online_envs = SubprocVecEnv(
                [
                    get_env_builder(i + 100, env_name=env_name, target_goal=target_goal)
                    for i in range(self.variant["num_online_rollouts"])
                ]
            )
            print("DEBUG: Online environments created successfully")
            
            print(f"DEBUG: Starting online tuning phase with {self.variant['max_online_iters']} iterations...")
            self.online_tuning(online_envs, eval_envs, loss_fn)
            print("DEBUG: Online tuning phase completed")
            
            print("DEBUG: Closing online environments...")
            online_envs.close()

        elif self.variant.get("render", False):
            print("\n" + "="*60)
            if self.variant.get("save_video", False):
                print("🎬 FINAL VIDEO RECORDING - Saving trained agent performance")
            else:
                print("🎬 FINAL RENDERING - Showing trained agent performance")
            print("="*60)
            
            if self.variant.get("save_video", False):
                final_render_outputs = self.render_evaluation_video()
            else:
                final_render_outputs = self.render_evaluation()
            print("\nTraining complete! Performance recorded.")
 
        else:
            print("DEBUG: Skipping online tuning phase (max_online_iters = 0)")

        print("DEBUG: Closing evaluation environments...")
        eval_envs.close()
        print("DEBUG: Experiment execution completed successfully!")


if __name__ == "__main__":
    print("DEBUG: Starting main execution...")
    print("DEBUG: All imports completed successfully!")
    parser = argparse.ArgumentParser()
    print("DEBUG: ArgumentParser created")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env", type=str, default="hopper-medium-v2")
    parser.add_argument("--model_type", type=str, choices=["dt", "dmamba", "dmamba-min"], default="dt", help="Choose model type: dt or dmamba")

    # model options
    parser.add_argument("--K", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--activation_function", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--eval_context_length", type=int, default=5)
    # 0: no pos embedding others: absolute ordering
    parser.add_argument("--ordering", type=int, default=0)

    # shared evaluation options
    parser.add_argument("--eval_rtg", type=int, default=3600)
    parser.add_argument("--num_eval_episodes", type=int, default=1)

    # shared training options
    parser.add_argument("--init_temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-wd", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=10000)

    # pretraining options
    parser.add_argument("--max_pretrain_iters", type=int, default=1)
    parser.add_argument("--num_updates_per_pretrain_iter", type=int, default=5000)

    # finetuning options
    parser.add_argument("--max_online_iters", type=int, default=1500)
    parser.add_argument("--online_rtg", type=int, default=7200)
    parser.add_argument("--num_online_rollouts", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=500)
    parser.add_argument("--num_updates_per_online_iter", type=int, default=300)
    parser.add_argument("--eval_interval", type=int, default=1)

    # environment options
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_to_tb", "-w", type=bool, default=True)
    parser.add_argument("--save_dir", type=str, default="./exp")
    parser.add_argument("--exp_name", type=str, default="default")
    
    # verbosity options
    parser.add_argument("--verbose", "-v", type=int, default=1, choices=[0, 1, 2], 
                       help="Verbosity level: 0=minimal, 1=normal, 2=detailed")

    parser.add_argument("--render", action="store_true", help="Enable rendering during evaluation")
    parser.add_argument("--render_episodes", type=int, default=1, help="Number of episodes to render")
    parser.add_argument("--render_interval", type=int, default=1, help="Render every N iterations during training")
    parser.add_argument("--render_fps", type=int, default=30, help="FPS for rendering")
    parser.add_argument("--save_video", action="store_true", help="Save videos instead of real-time rendering")
    parser.add_argument("--video_episodes", type=int, default=1, help="Number of video episodes to save")



    args = parser.parse_args()
    print("DEBUG: Arguments parsed successfully")
    
    # Auto-detect Apple Silicon GPU (MPS) if requested
    if args.device.lower() in ["gpu", "mps"]:
        try:
            import torch
            if torch.backends.mps.is_available():
                args.device = "mps"
                print("INFO: Using Apple Silicon MPS backend for computation.")
            else:
                print("WARNING: MPS backend not available, falling back to CPU.")
                args.device = "cpu"
        except Exception as e:
            print(f"WARNING: Could not check MPS backend ({e}), falling back to CPU.")
            args.device = "cpu"

    # Set global verbosity
    VERBOSE_LEVEL = args.verbose
    
    def vprint(level, message):
        """Verbose print function"""
        if VERBOSE_LEVEL >= level:
            print(message)

    vprint(1, f"=== Online Decision Transformer Training (Verbosity Level: {VERBOSE_LEVEL}) ===")
    vprint(2, f"Arguments: {vars(args)}")

    utils.set_seed_everywhere(args.seed)
    print("DEBUG: Seed set successfully")
    
    vprint(1, f"✓ Random seed set to {args.seed}")
    vprint(2, f"✓ Environment: {args.env}")
    vprint(2, f"✓ Model type: {args.model_type}")
    vprint(2, f"✓ Device: {args.device}")
    vprint(2, f"✓ Pretrain iterations: {args.max_pretrain_iters}")
    vprint(2, f"✓ Online iterations: {args.max_online_iters}")
    
    experiment = Experiment(vars(args))
    print("DEBUG: Experiment object created successfully")

    print("=" * 50)
    print("DEBUG: About to call experiment()")
    vprint(1, "🚀 Starting experiment execution...")
    experiment()
"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
from tqdm import tqdm
import time
import gymnasium as gym
import os
from datetime import datetime
import imageio



MAX_EPISODE_LEN = 1000  # Reduced for faster testing


def create_vec_eval_episodes_fn(
    vec_env,
    eval_rtg,
    state_dim,
    act_dim,
    state_mean,
    state_std,
    device,
    use_mean=False,
    reward_scale=0.001,
    verbose=1,
):
    def eval_episodes_fn(model):
        if verbose >= 1:
            print(f"🎯 Running evaluation with {vec_env.num_envs} environments...")
            
        target_return = [eval_rtg * reward_scale] * vec_env.num_envs
        
        eval_start_time = time.time() if verbose >= 2 else None
        
        returns, lengths, _ = vec_evaluate_episode_rtg(
            vec_env,
            state_dim,
            act_dim,
            model,
            max_ep_len=MAX_EPISODE_LEN,
            reward_scale=reward_scale,
            target_return=target_return,
            mode="normal",
            state_mean=state_mean,
            state_std=state_std,
            device=device,
            use_mean=use_mean,
            verbose=verbose,
        )
        
        if verbose >= 2:
            eval_time = time.time() - eval_start_time
            print(f"📈 Evaluation Results:")
            print(f"  • Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
            print(f"  • Mean Length: {np.mean(lengths):.1f} ± {np.std(lengths):.1f}")
            print(f"  • Evaluation Time: {eval_time:.2f}s")
        elif verbose >= 1:
            print(f"✓ Evaluation completed: Return {np.mean(returns):.2f}, Length {np.mean(lengths):.1f}")
            
        suffix = "_gm" if use_mean else ""
        return {
            f"evaluation/return_mean{suffix}": np.mean(returns),
            f"evaluation/return_std{suffix}": np.std(returns),
            f"evaluation/length_mean{suffix}": np.mean(lengths),
            f"evaluation/length_std{suffix}": np.std(lengths),
        }

    return eval_episodes_fn


@torch.no_grad()
def vec_evaluate_episode_rtg(
    vec_env,
    state_dim,
    act_dim,
    model,
    target_return: list,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    mode="normal",
    use_mean=False,
    verbose=1,
):
    assert len(target_return) == vec_env.num_envs

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    num_envs = vec_env.num_envs
    state = vec_env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = (
        torch.from_numpy(state)
        .reshape(num_envs, state_dim)
        .to(device=device, dtype=torch.float32)
    ).reshape(num_envs, -1, state_dim)
    actions = torch.zeros(0, device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(
        num_envs, -1, 1
    )
    timesteps = torch.tensor([0] * num_envs, device=device, dtype=torch.long).reshape(
        num_envs, -1
    )

    # episode_return, episode_length = 0.0, 0
    episode_return = np.zeros((num_envs, 1)).astype(float)
    episode_length = np.full(num_envs, np.inf)

    unfinished = np.ones(num_envs).astype(bool)
    
    # Create progress bar for evaluation steps
    if verbose >= 2:
        pbar = tqdm(range(max_ep_len), desc="Evaluation Steps", leave=False, dynamic_ncols=True)
    else:
        pbar = range(max_ep_len)
        
    for t in pbar:
        # add padding
        actions = torch.cat(
            [
                actions,
                torch.zeros((num_envs, act_dim), device=device).reshape(
                    num_envs, -1, act_dim
                ),
            ],
            dim=1,
        )
        rewards = torch.cat(
            [
                rewards,
                torch.zeros((num_envs, 1), device=device).reshape(num_envs, -1, 1),
            ],
            dim=1,
        )

        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=num_envs,
        )
        state_pred = state_pred.detach().cpu().numpy().reshape(num_envs, -1)
        reward_pred = reward_pred.detach().cpu().numpy().reshape(num_envs)

        # the return action is a SquashNormal distribution
        action = action_dist.sample().reshape(num_envs, -1, act_dim)[:, -1]
        if use_mean:
            action = action_dist.mean.reshape(num_envs, -1, act_dim)[:, -1]
        action = action.clamp(*model.action_range)

        state, reward, done, _ = vec_env.step(action.detach().cpu().numpy())

        # eval_env.step() will execute the action for all the sub-envs, for those where
        # the episodes have terminated, the envs will be reset. Hence we use
        # "unfinished" to track whether the first episode we roll out for each sub-env is
        # finished. In contrast, "done" only relates to the current episode
        episode_return[unfinished] += reward[unfinished].reshape(-1, 1)

        actions[:, -1] = action
        state = (
            torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(num_envs, -1, state_dim)
        )
        states = torch.cat([states, state], dim=1)
        reward = torch.from_numpy(reward.astype(np.float32)).to(device=device, dtype=torch.float32).reshape(num_envs, 1)
        rewards[:, -1] = reward

        if mode != "delayed":
            pred_return = target_return[:, -1] - (reward * reward_scale)
        else:
            pred_return = target_return[:, -1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(num_envs, -1, 1)], dim=1
        )

        timesteps = torch.cat(
            [
                timesteps,
                torch.ones((num_envs, 1), device=device, dtype=torch.long).reshape(
                    num_envs, 1
                )
                * (t + 1),
            ],
            dim=1,
        )

        if t == max_ep_len - 1:
            done = np.ones(done.shape).astype(bool)

        if np.any(done):
            ind = np.where(done)[0]
            unfinished[ind] = False
            episode_length[ind] = np.minimum(episode_length[ind], t + 1)

        if not np.any(unfinished):
            if verbose >= 2 and hasattr(pbar, 'set_description'):
                pbar.set_description(f"Evaluation Steps (all episodes finished at step {t+1})")
            break

        # Update progress bar with current status
        if verbose >= 2 and hasattr(pbar, 'set_postfix'):
            active_envs = np.sum(unfinished)
            avg_return = np.mean(episode_return[~unfinished]) if np.any(~unfinished) else 0
            pbar.set_postfix({
                'active_envs': active_envs,
                'avg_return': f'{avg_return:.1f}',
                'step': t+1
            })

    trajectories = []
    for ii in range(num_envs):
        ep_len = episode_length[ii].astype(int)
        terminals = np.zeros(ep_len)
        terminals[-1] = 1
        traj = {
            "observations": states[ii].detach().cpu().numpy()[:ep_len],
            "actions": actions[ii].detach().cpu().numpy()[:ep_len],
            "rewards": rewards[ii].detach().cpu().numpy()[:ep_len],
            "terminals": terminals,
        }
        trajectories.append(traj)

    return (
        episode_return.reshape(num_envs),
        episode_length.reshape(num_envs),
        trajectories,
    )



@torch.no_grad()
def render_episode(
    env_name,
    model,
    state_dim,
    act_dim,
    target_return,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    use_mean=True,
    render_mode="human",
    fps=30,
):
    """
    Render a single episode with the trained model.
    This uses the base Mujoco environment which supports rendering.
    """
    model.eval()
    model.to(device=device)
    
    # Create base Mujoco environment (not D4RL) for better rendering support
    # Map D4RL env names to base Mujoco environments
    env_mapping = {
        "hopper": "Hopper-v4",  # Updated to v4
        "walker2d": "Walker2d-v4", 
        "halfcheetah": "HalfCheetah-v4",
        "ant": "Ant-v4",
    }
    
    # Try to create appropriate environment
    render_env_name = env_name
    for d4rl_prefix, mujoco_name in env_mapping.items():
        if d4rl_prefix in env_name.lower():
            render_env_name = mujoco_name
            print(f"Using {mujoco_name} for rendering (mapped from {env_name})")
            break
    
    try:
        # Try creating the Mujoco environment first
        env = gym.make(render_env_name, render_mode=render_mode)
    except:
        # Fallback to original D4RL environment
        print(f"Could not create {render_env_name}, falling back to {env_name}")
        try:
            env = gym.make(env_name)
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    
    # Reset environment - handle both old and new gym API
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        state = reset_output[0]
    else:
        state = reset_output
    
    # Initialize history tensors
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    states = states.reshape(1, -1, state_dim)
    actions = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, -1, 1)
    timesteps = torch.tensor([0], device=device, dtype=torch.long).reshape(1, -1)
    
    episode_return = 0.0
    episode_length = 0
    
    print(f"Starting rendering episode with target return: {target_return.item():.2f}")
    
    for t in range(max_ep_len):
        # Add padding
        actions = torch.cat([actions, torch.zeros((1, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((1, 1, 1), device=device)], dim=1)
        
        # Get model predictions
        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=1,
        )
        
        # Sample action
        if use_mean:
            action = action_dist.mean.reshape(1, -1, act_dim)[:, -1]
        else:
            action = action_dist.sample().reshape(1, -1, act_dim)[:, -1]
        action = action.clamp(*model.action_range)
        
        # Execute action - handle both old and new gym API
        step_output = env.step(action.detach().cpu().numpy().squeeze())
        
        # Handle different gym API versions
        if len(step_output) == 4:
            # Old gym API: (obs, reward, done, info)
            next_state, reward, done, info = step_output
        elif len(step_output) == 5:
            # New gymnasium API: (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, info = step_output
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected number of values from env.step(): {len(step_output)}")
        
        # Render
        try:
            if hasattr(env, 'render'):
                env.render()
            if render_mode == "human":
                time.sleep(1.0 / fps)  # Control rendering speed
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
        
        episode_return += reward
        episode_length += 1
        
        # Update history
        actions[:, -1] = action
        next_state_tensor = torch.from_numpy(next_state).to(device=device, dtype=torch.float32)
        states = torch.cat([states, next_state_tensor.reshape(1, -1, state_dim)], dim=1)
        rewards[:, -1] = torch.tensor(reward, device=device, dtype=torch.float32).reshape(1, 1)
        
        # Update target return
        pred_return = target_return[:, -1] - (rewards[:, -1] * reward_scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, -1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
        
        if done:
            break
    
    env.close()
    
    print(f"Episode finished: Return = {episode_return:.2f}, Length = {episode_length}")
    return episode_return, episode_length

@torch.no_grad()
def render_episode_video(
    env_name,
    model,
    state_dim,
    act_dim,
    target_return,
    max_ep_len=1000,
    reward_scale=0.001,
    state_mean=0.0,
    state_std=1.0,
    device="cuda",
    use_mean=True,
    video_folder="./videos",
    fps=30,
):
    """
    Render a single episode and save it as a video file.
    """
    model.eval()
    model.to(device=device)
    
    # Create video folder if it doesn't exist
    os.makedirs(video_folder, exist_ok=True)
    
    # Create base Mujoco environment for rendering
    env_mapping = {
        "hopper": "Hopper-v4",
        "walker2d": "Walker2d-v4", 
        "halfcheetah": "HalfCheetah-v4",
        "ant": "Ant-v4",
    }
    
    render_env_name = env_name
    for d4rl_prefix, mujoco_name in env_mapping.items():
        if d4rl_prefix in env_name.lower():
            render_env_name = mujoco_name
            print(f"Using {mujoco_name} for video recording (mapped from {env_name})")
            break
    
    try:
        # Create environment with rgb_array mode for video recording
        env = gym.make(render_env_name, render_mode="rgb_array")
    except:
        print(f"Could not create {render_env_name}, falling back to {env_name}")
        try:
            env = gym.make(env_name)
        except Exception as e:
            print(f"Error creating environment: {e}")
            raise
    
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)
    
    # Reset environment
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        state = reset_output[0]
    else:
        state = reset_output
    
    # Initialize history tensors
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    states = states.reshape(1, -1, state_dim)
    actions = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((1, 0, 1), device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, -1, 1)
    timesteps = torch.tensor([0], device=device, dtype=torch.long).reshape(1, -1)
    
    episode_return = 0.0
    episode_length = 0
    frames = []
    
    print(f"Recording episode with target return: {target_return.item():.2f}")
    
    # Capture first frame
    try:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    except:
        print("Warning: Could not capture first frame")
    
    for t in range(max_ep_len):
        # Add padding
        actions = torch.cat([actions, torch.zeros((1, 1, act_dim), device=device)], dim=1)
        rewards = torch.cat([rewards, torch.zeros((1, 1, 1), device=device)], dim=1)
        
        # Get model predictions
        state_pred, action_dist, reward_pred = model.get_predictions(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            num_envs=1,
        )
        
        # Sample action
        if use_mean:
            action = action_dist.mean.reshape(1, -1, act_dim)[:, -1]
        else:
            action = action_dist.sample().reshape(1, -1, act_dim)[:, -1]
        action = action.clamp(*model.action_range)
        
        # Execute action
        step_output = env.step(action.detach().cpu().numpy().squeeze())
        
        # Handle different gym API versions
        if len(step_output) == 4:
            next_state, reward, done, info = step_output
        elif len(step_output) == 5:
            next_state, reward, terminated, truncated, info = step_output
            done = terminated or truncated
        else:
            raise ValueError(f"Unexpected number of values from env.step(): {len(step_output)}")
        
        # Capture frame
        try:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        except Exception as e:
            print(f"Warning: Could not capture frame at step {t}: {e}")
        
        episode_return += reward
        episode_length += 1
        
        # Update history
        actions[:, -1] = action
        next_state_tensor = torch.from_numpy(next_state).to(device=device, dtype=torch.float32)
        states = torch.cat([states, next_state_tensor.reshape(1, -1, state_dim)], dim=1)
        rewards[:, -1] = torch.tensor(reward, device=device, dtype=torch.float32).reshape(1, 1)
        
        # Update target return
        pred_return = target_return[:, -1] - (rewards[:, -1] * reward_scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, -1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)
        
        if done:
            break
    
    env.close()
    
    # Save video
    if len(frames) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"{video_folder}/{env_name}_return{episode_return:.0f}_len{episode_length}_{timestamp}.mp4"
        
        print(f"Saving video with {len(frames)} frames to {video_filename}")
        try:
            imageio.mimsave(video_filename, frames, fps=fps)
            print(f"✅ Video saved successfully: {video_filename}")
        except Exception as e:
            print(f"Error saving video: {e}")
            # Try alternative format
            try:
                video_filename = video_filename.replace('.mp4', '.gif')
                imageio.mimsave(video_filename, frames, fps=fps)
                print(f"✅ Video saved as GIF: {video_filename}")
            except:
                print("Failed to save video in any format")
    else:
        print("No frames captured, video not saved")
    
    print(f"Episode finished: Return = {episode_return:.2f}, Length = {episode_length}")
    return episode_return, episode_length
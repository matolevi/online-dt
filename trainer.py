"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

import numpy as np
import torch
import time
from tqdm import tqdm


class SequenceTrainer:
    def __init__(
        self,
        model,
        optimizer,
        log_temperature_optimizer,
        scheduler=None,
        device="cuda",
        verbose=1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.log_temperature_optimizer = log_temperature_optimizer
        self.scheduler = scheduler
        self.device = device
        self.start_time = time.time()
        self.verbose = verbose

    def train_iteration(
        self,
        loss_fn,
        dataloader,
    ):
        losses, nlls, entropies = [], [], []
        logs = dict()
        train_start = time.time()

        self.model.train()
        
        # Create progress bar based on verbosity
        if self.verbose >= 1:
            pbar = tqdm(enumerate(dataloader), total=len(dataloader), 
                       desc="Training", leave=False, dynamic_ncols=True)
            if self.verbose >= 2:
                pbar.set_postfix_str("Initializing...")
        else:
            pbar = enumerate(dataloader)
            
        step_start_time = time.time()
        for step_idx, trajs in pbar:
            loss, nll, entropy = self.train_step_stochastic(loss_fn, trajs)
            losses.append(loss)
            nlls.append(nll)
            entropies.append(entropy)
            
            # Update progress bar with detailed info
            if self.verbose >= 1 and hasattr(pbar, 'set_postfix'):
                current_lr = self.optimizer.param_groups[0]['lr'] if self.optimizer.param_groups else 0
                temp_value = self.model.temperature().detach().cpu().item()
                
                if self.verbose >= 2:
                    step_time = time.time() - step_start_time
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'entropy': f'{entropy:.3f}', 
                        'temp': f'{temp_value:.3f}',
                        'lr': f'{current_lr:.2e}',
                        'step_time': f'{step_time:.2f}s'
                    })
                else:
                    pbar.set_postfix({
                        'loss': f'{loss:.4f}',
                        'temp': f'{temp_value:.3f}'
                    })
                
                # Print periodic updates for detailed verbosity
                if self.verbose >= 2 and (step_idx + 1) % 100 == 0:
                    avg_loss = np.mean(losses[-100:])
                    print(f"\n  Step {step_idx + 1}/{len(dataloader)}: Avg Loss (last 100): {avg_loss:.4f}")
                    
            step_start_time = time.time()

        if self.verbose >= 1:
            print(f"\n✓ Training iteration completed in {time.time() - train_start:.2f}s")

        logs["time/training"] = time.time() - train_start
        logs["training/train_loss_mean"] = np.mean(losses)
        logs["training/train_loss_std"] = np.std(losses)
        logs["training/nll"] = nlls[-1]
        logs["training/entropy"] = entropies[-1]
        logs["training/temp_value"] = self.model.temperature().detach().cpu().item()
        
        if self.verbose >= 2:
            print(f"📊 Training Statistics:")
            print(f"  • Mean Loss: {logs['training/train_loss_mean']:.4f} ± {logs['training/train_loss_std']:.4f}")
            print(f"  • Final NLL: {logs['training/nll']:.4f}")
            print(f"  • Final Entropy: {logs['training/entropy']:.4f}")
            print(f"  • Temperature: {logs['training/temp_value']:.4f}")
            print(f"  • Training Time: {logs['time/training']:.2f}s")

        return logs

    def train_step_stochastic(self, loss_fn, trajs):
        (
            states,
            actions,
            rewards,
            dones,
            rtg,
            timesteps,
            ordering,
            padding_mask,
        ) = trajs

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        dones = dones.to(self.device)
        rtg = rtg.to(self.device)
        timesteps = timesteps.to(self.device)
        ordering = ordering.to(self.device)
        padding_mask = padding_mask.to(self.device)

        action_target = torch.clone(actions)

        _, action_preds, _ = self.model.forward(
            states,
            actions,
            rewards,
            rtg[:, :-1],
            timesteps,
            ordering,
            padding_mask=padding_mask,
        )

        loss, nll, entropy = loss_fn(
            action_preds,  # a_hat_dist
            action_target,
            padding_mask,
            self.model.temperature().detach(),  # no gradient taken here
        )
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
        self.optimizer.step()

        # Only optimize temperature for stochastic models
        if self.log_temperature_optimizer is not None:
            self.log_temperature_optimizer.zero_grad()
            temperature_loss = (
                self.model.temperature() * (entropy - self.model.target_entropy).detach()
            )
            temperature_loss.backward()
            self.log_temperature_optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        return (
            loss.detach().cpu().item(),
            nll.detach().cpu().item(),
            entropy.detach().cpu().item(),
        )

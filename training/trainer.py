"""
PPO (Proximal Policy Optimization) trainer for poker agents.

Implements:
- PPO algorithm for policy optimization
- Value function training
- Checkpoint management
- Training orchestration
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time

from training.model import PokerTransformer, PokerTokenizer
from training.data_collector import DataCollector, ReplayBuffer
from training.metrics import MetricsLogger
from training.training_config import FullConfig


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for poker agents.
    
    Implements the PPO algorithm with:
    - Clipped surrogate objective
    - Value function learning
    - Entropy regularization
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: PokerTransformer,
        config: FullConfig,
        logger: MetricsLogger,
        device: str = "cpu"
    ):
        """
        Initialize PPO trainer.
        
        Args:
            model: Poker transformer model
            config: Full training configuration
            logger: Metrics logger
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.logger = logger
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.num_games // config.training.games_per_update
        )
        
        # Training state
        self.global_step = 0
        self.total_episodes = 0
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards tensor (batch_size,)
            values: Value estimates (batch_size,)
            dones: Done flags (batch_size,)
            next_values: Next state values (batch_size,)
        
        Returns:
            Tuple of (advantages, returns)
        """
        gamma = self.config.training.gamma
        gae_lambda = self.config.training.gae_lambda
        
        batch_size = rewards.size(0)
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        gae = 0.0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_value = next_values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor
    ) -> Tuple[float, float, float]:
        """
        Perform PPO update.
        
        Args:
            states: State tensors
            actions: Action indices
            old_log_probs: Old log probabilities
            advantages: Advantage estimates
            returns: Return estimates
        
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Multiple epochs over the same data
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0
        
        for _ in range(self.config.training.ppo_epochs):
            # Shuffle data
            indices = torch.randperm(states.size(0))
            
            # Mini-batch updates
            batch_size = states.size(0) // self.config.training.num_minibatches
            
            for i in range(self.config.training.num_minibatches):
                start = i * batch_size
                end = start + batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass
                new_log_probs, values, entropy = self.model.evaluate_actions(
                    mb_states, mb_actions
                )
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.training.clip_epsilon,
                    1.0 + self.config.training.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, mb_returns)
                
                # Total loss
                loss = (
                    policy_loss +
                    self.config.training.value_loss_coef * value_loss -
                    self.config.training.entropy_coef * entropy.mean()
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
                )
                
                self.optimizer.step()
                
                # Accumulate losses
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1
        
        # Average losses
        avg_policy_loss = total_policy_loss / num_updates
        avg_value_loss = total_value_loss / num_updates
        avg_entropy = total_entropy / num_updates
        
        return avg_policy_loss, avg_value_loss, avg_entropy
    
    def train_step(
        self,
        data_collector: DataCollector
    ) -> Optional[dict]:
        """
        Perform one training step.
        
        Args:
            data_collector: Data collector with replay buffer
        
        Returns:
            Dictionary of training metrics or None if not enough data
        """
        # Get batch from replay buffer
        batch = data_collector.get_batch(self.config.training.batch_size)
        if batch is None:
            return None
        
        states, actions, rewards, next_states, dones, old_log_probs, old_values = batch
        
        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        old_values = old_values.to(self.device)
        
        # Compute next state values
        with torch.no_grad():
            next_values = self.model.get_value(next_states).squeeze(-1)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            rewards, old_values, dones, next_values
        )
        
        # PPO update
        policy_loss, value_loss, entropy = self.ppo_update(
            states, actions, old_log_probs, advantages, returns
        )
        
        # Update learning rate
        self.scheduler.step()
        current_lr = self.scheduler.get_last_lr()[0]
        
        # Compute gradient norm
        grad_norm = sum(
            p.grad.norm().item() ** 2 for p in self.model.parameters() if p.grad is not None
        ) ** 0.5
        
        # Log metrics
        self.logger.log_training_step(
            step=self.global_step,
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            lr=current_lr,
            grad_norm=grad_norm
        )
        
        self.global_step += 1
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'lr': current_lr,
            'grad_norm': grad_norm
        }
    
    def save_checkpoint(self, path: Path, **kwargs) -> None:
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            **kwargs: Additional data to save
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'total_episodes': self.total_episodes,
            **kwargs
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Path) -> dict:
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Dictionary with additional checkpoint data
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.total_episodes = checkpoint['total_episodes']
        
        print(f"Checkpoint loaded from {path}")
        print(f"Resuming from step {self.global_step}, episode {self.total_episodes}")
        
        return checkpoint


if __name__ == "__main__":
    # Test trainer initialization
    from training.training_config import get_default_config
    
    config = get_default_config()
    tokenizer = PokerTokenizer()
    
    model = PokerTransformer(
        state_dim=tokenizer.get_state_dim(),
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        d_ff=config.model.d_ff,
        num_actions=config.model.num_actions
    )
    
    logger = MetricsLogger(config.paths.log_dir, use_tensorboard=False)
    trainer = PPOTrainer(model, config, logger)
    
    print(f"Trainer initialized with {model.count_parameters():,} parameters")

"""
Training metrics and logging utilities.

Tracks:
- Episode rewards and returns
- Policy entropy
- Value function accuracy
- Poker-specific metrics (VPIP, PFR, aggression)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
from collections import defaultdict
import json
from pathlib import Path


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""
    
    # Training metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    policy_loss: List[float] = field(default_factory=list)
    value_loss: List[float] = field(default_factory=list)
    entropy: List[float] = field(default_factory=list)
    
    # Poker-specific metrics
    vpip: List[float] = field(default_factory=list)  # Voluntarily put $ in pot
    pfr: List[float] = field(default_factory=list)  # Pre-flop raise
    aggression_factor: List[float] = field(default_factory=list)
    win_rate: List[float] = field(default_factory=list)
    
    # Learning metrics
    learning_rate: List[float] = field(default_factory=list)
    grad_norm: List[float] = field(default_factory=list)
    
    def add_episode(
        self,
        reward: float,
        length: int,
        vpip: float = 0.0,
        pfr: float = 0.0,
        aggression: float = 0.0,
        won: bool = False
    ) -> None:
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.vpip.append(vpip)
        self.pfr.append(pfr)
        self.aggression_factor.append(aggression)
        self.win_rate.append(1.0 if won else 0.0)
    
    def add_training_step(
        self,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        lr: float,
        grad_norm: float
    ) -> None:
        """Add training step metrics."""
        self.policy_loss.append(policy_loss)
        self.value_loss.append(value_loss)
        self.entropy.append(entropy)
        self.learning_rate.append(lr)
        self.grad_norm.append(grad_norm)
    
    def get_recent_mean(self, metric_name: str, window: int = 100) -> float:
        """Get mean of recent values for a metric."""
        values = getattr(self, metric_name, [])
        if not values:
            return 0.0
        recent = values[-window:]
        return sum(recent) / len(recent)
    
    def save(self, path: Path) -> None:
        """Save metrics to JSON file."""
        data = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'entropy': self.entropy,
            'vpip': self.vpip,
            'pfr': self.pfr,
            'aggression_factor': self.aggression_factor,
            'win_rate': self.win_rate,
            'learning_rate': self.learning_rate,
            'grad_norm': self.grad_norm
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingMetrics':
        """Load metrics from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class MetricsLogger:
    """Logger for training metrics with TensorBoard support."""
    
    def __init__(self, log_dir: Path, use_tensorboard: bool = True):
        """
        Initialize metrics logger.
        
        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = TrainingMetrics()
        self.use_tensorboard = use_tensorboard
        self.writer = None
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("TensorBoard not available, logging to JSON only")
                self.use_tensorboard = False
    
    def log_episode(
        self,
        episode_num: int,
        reward: float,
        length: int,
        **kwargs
    ) -> None:
        """
        Log episode metrics.
        
        Args:
            episode_num: Episode number
            reward: Total episode reward
            length: Episode length
            **kwargs: Additional metrics (vpip, pfr, etc.)
        """
        self.metrics.add_episode(
            reward=reward,
            length=length,
            vpip=kwargs.get('vpip', 0.0),
            pfr=kwargs.get('pfr', 0.0),
            aggression=kwargs.get('aggression', 0.0),
            won=kwargs.get('won', False)
        )
        
        if self.writer:
            self.writer.add_scalar('Episode/Reward', reward, episode_num)
            self.writer.add_scalar('Episode/Length', length, episode_num)
            for key, value in kwargs.items():
                self.writer.add_scalar(f'Episode/{key}', value, episode_num)
    
    def log_training_step(
        self,
        step: int,
        policy_loss: float,
        value_loss: float,
        entropy: float,
        lr: float,
        grad_norm: float
    ) -> None:
        """Log training step metrics."""
        self.metrics.add_training_step(
            policy_loss=policy_loss,
            value_loss=value_loss,
            entropy=entropy,
            lr=lr,
            grad_norm=grad_norm
        )
        
        if self.writer:
            self.writer.add_scalar('Train/PolicyLoss', policy_loss, step)
            self.writer.add_scalar('Train/ValueLoss', value_loss, step)
            self.writer.add_scalar('Train/Entropy', entropy, step)
            self.writer.add_scalar('Train/LearningRate', lr, step)
            self.writer.add_scalar('Train/GradNorm', grad_norm, step)
    
    def log_evaluation(
        self,
        step: int,
        win_rate: float,
        avg_reward: float,
        **kwargs
    ) -> None:
        """Log evaluation metrics."""
        if self.writer:
            self.writer.add_scalar('Eval/WinRate', win_rate, step)
            self.writer.add_scalar('Eval/AvgReward', avg_reward, step)
            for key, value in kwargs.items():
                self.writer.add_scalar(f'Eval/{key}', value, step)
    
    def print_summary(self, window: int = 100) -> None:
        """Print summary of recent metrics."""
        print("\n" + "=" * 50)
        print("Training Metrics Summary (last {} episodes)".format(window))
        print("=" * 50)
        print(f"Avg Reward: {self.metrics.get_recent_mean('episode_rewards', window):.2f}")
        print(f"Avg Length: {self.metrics.get_recent_mean('episode_lengths', window):.1f}")
        print(f"Win Rate: {self.metrics.get_recent_mean('win_rate', window):.2%}")
        print(f"VPIP: {self.metrics.get_recent_mean('vpip', window):.2%}")
        print(f"PFR: {self.metrics.get_recent_mean('pfr', window):.2%}")
        print(f"Aggression: {self.metrics.get_recent_mean('aggression_factor', window):.2f}")
        
        if self.metrics.policy_loss:
            print(f"\nPolicy Loss: {self.metrics.get_recent_mean('policy_loss', window):.4f}")
            print(f"Value Loss: {self.metrics.get_recent_mean('value_loss', window):.4f}")
            print(f"Entropy: {self.metrics.get_recent_mean('entropy', window):.4f}")
        print("=" * 50 + "\n")
    
    def save(self, path: Optional[Path] = None) -> None:
        """Save metrics to file."""
        if path is None:
            path = self.log_dir / "metrics.json"
        self.metrics.save(path)
    
    def close(self) -> None:
        """Close logger and save metrics."""
        self.save()
        if self.writer:
            self.writer.close()


class PokerMetricsCalculator:
    """Calculate poker-specific metrics from game data."""
    
    @staticmethod
    def calculate_vpip(actions: List[str]) -> float:
        """
        Calculate VPIP (Voluntarily Put money In Pot).
        
        Percentage of hands where player voluntarily put money in pot.
        """
        if not actions:
            return 0.0
        
        voluntary_actions = ['call', 'raise', 'all_in']
        vpip_count = sum(1 for a in actions if a.lower() in voluntary_actions)
        return vpip_count / len(actions)
    
    @staticmethod
    def calculate_pfr(preflop_actions: List[str]) -> float:
        """
        Calculate PFR (Pre-Flop Raise).
        
        Percentage of hands where player raised pre-flop.
        """
        if not preflop_actions:
            return 0.0
        
        raise_count = sum(1 for a in preflop_actions if a.lower() in ['raise', 'all_in'])
        return raise_count / len(preflop_actions)
    
    @staticmethod
    def calculate_aggression_factor(
        bets_and_raises: int,
        calls: int
    ) -> float:
        """
        Calculate aggression factor.
        
        (Bets + Raises) / Calls
        """
        if calls == 0:
            return float(bets_and_raises) if bets_and_raises > 0 else 0.0
        return bets_and_raises / calls


if __name__ == "__main__":
    # Test metrics logger
    logger = MetricsLogger(Path("training/logs/test"), use_tensorboard=False)
    
    # Log some episodes
    for i in range(10):
        logger.log_episode(
            episode_num=i,
            reward=100.0 * (i + 1),
            length=50,
            vpip=0.3,
            pfr=0.2,
            won=(i % 2 == 0)
        )
    
    logger.print_summary()
    logger.close()
    print("Metrics logged successfully")

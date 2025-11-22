"""
Training configuration for LLM poker agents.

This module defines all configuration dataclasses for:
- Model architecture parameters
- Training hyperparameters
- Simulation settings
- Logging and checkpoint paths
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for the transformer-based poker LLM."""
    
    # Architecture
    d_model: int = 256  # Embedding dimension
    n_heads: int = 8  # Number of attention heads
    n_layers: int = 6  # Number of transformer layers
    d_ff: int = 1024  # Feed-forward dimension
    dropout: float = 0.1
    
    # Input/Output
    max_seq_len: int = 128  # Maximum sequence length for history
    num_actions: int = 5  # fold, check, call, raise, all-in
    
    # Card encoding
    num_cards: int = 52  # Standard deck
    num_positions: int = 9  # Max players
    
    def total_params(self) -> int:
        """Estimate total model parameters."""
        # Rough estimate: embeddings + layers + heads
        embed_params = self.d_model * (self.num_cards + self.num_positions + 100)
        layer_params = self.n_layers * (
            4 * self.d_model * self.d_model +  # Attention
            2 * self.d_model * self.d_ff  # FFN
        )
        head_params = self.d_model * (self.num_actions + 1)  # Action + value heads
        return embed_params + layer_params + head_params


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    
    # Training
    num_games: int = 1_000_000  # Total games to play
    batch_size: int = 256  # Batch size for training
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE parameter
    clip_epsilon: float = 0.2  # PPO clipping
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # PPO specific
    ppo_epochs: int = 4  # Epochs per batch
    num_minibatches: int = 4
    
    # Simulation
    parallel_games: int = 8  # Number of games to run in parallel
    games_per_update: int = 128  # Collect this many games before update
    
    # Evaluation
    eval_frequency: int = 1000  # Evaluate every N updates
    eval_games: int = 100  # Number of games for evaluation
    
    # Checkpointing
    checkpoint_frequency: int = 5000  # Save every N updates
    keep_checkpoints: int = 5  # Number of checkpoints to keep
    
    # Logging
    log_frequency: int = 10  # Log every N updates
    
    # Device
    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 42


@dataclass
class SimulationConfig:
    """Game simulation settings."""
    
    # Game settings
    num_players: int = 6
    starting_chips: int = 1000
    small_blind: int = 5
    big_blind: int = 10
    max_hands: int = 100
    
    # Performance
    use_multiprocessing: bool = False  # Disabled by default due to model pickling
    num_workers: Optional[int] = None  # None = auto-detect
    
    # Data collection
    collect_all_states: bool = True  # Collect all states or just terminal
    augment_data: bool = True  # Use data augmentation


@dataclass
class PathConfig:
    """File paths for checkpoints, logs, and data."""
    
    # Base directories
    base_dir: Path = field(default_factory=lambda: Path("training"))
    checkpoint_dir: Path = field(default_factory=lambda: Path("training/checkpoints"))
    log_dir: Path = field(default_factory=lambda: Path("training/logs"))
    data_dir: Path = field(default_factory=lambda: Path("training/data"))
    
    # Config file
    config_file: Path = field(default_factory=lambda: Path("config/training_config.yaml"))
    
    def create_dirs(self) -> None:
        """Create all necessary directories."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class FullConfig:
    """Complete configuration combining all sub-configs."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.model.d_model % self.model.n_heads == 0, \
            "d_model must be divisible by n_heads"
        assert self.training.batch_size % self.training.num_minibatches == 0, \
            "batch_size must be divisible by num_minibatches"
        
        # Create directories
        self.paths.create_dirs()
    
    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        import yaml
        from dataclasses import asdict
        
        config_dict = asdict(self)
        # Convert Path objects to strings
        config_dict['paths'] = {k: str(v) for k, v in config_dict['paths'].items()}
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path) -> 'FullConfig':
        """Load configuration from YAML file."""
        import yaml
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert string paths back to Path objects
        if 'paths' in config_dict:
            config_dict['paths'] = {k: Path(v) for k, v in config_dict['paths'].items()}
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            simulation=SimulationConfig(**config_dict.get('simulation', {})),
            paths=PathConfig(**config_dict.get('paths', {}))
        )


def get_default_config() -> FullConfig:
    """Get default configuration for training."""
    return FullConfig()


if __name__ == "__main__":
    # Print default config info
    config = get_default_config()
    print("Default Training Configuration")
    print("=" * 50)
    print(f"Model Parameters: ~{config.model.total_params():,}")
    print(f"Total Games: {config.training.num_games:,}")
    print(f"Parallel Games: {config.training.parallel_games}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Learning Rate: {config.training.learning_rate}")
    print(f"Device: {config.training.device}")

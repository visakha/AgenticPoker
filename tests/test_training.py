"""Integration tests for training pipeline."""

import pytest
import torch
from pathlib import Path
import tempfile

from training.training_config import FullConfig, ModelConfig, TrainingConfig, SimulationConfig, PathConfig
from training.model import PokerTransformer, PokerTokenizer
from training.trainer import PPOTrainer
from training.metrics import MetricsLogger
from training.data_collector import DataCollector, ReplayBuffer, Experience


class TestTrainingPipeline:
    """Test training pipeline integration."""
    
    def test_config_creation(self):
        """Test configuration can be created."""
        config = FullConfig()
        assert config is not None
        assert config.model is not None
        assert config.training is not None
    
    def test_config_save_load(self):
        """Test configuration save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FullConfig()
            path = Path(tmpdir) / "config.yaml"
            
            config.save(path)
            assert path.exists()
            
            loaded_config = FullConfig.load(path)
            assert loaded_config.model.d_model == config.model.d_model
    
    def test_data_collector(self):
        """Test data collector."""
        tokenizer = PokerTokenizer()
        buffer = ReplayBuffer(capacity=100)
        collector = DataCollector(tokenizer, buffer)
        
        assert collector is not None
        assert len(buffer) == 0
    
    def test_replay_buffer(self):
        """Test replay buffer operations."""
        buffer = ReplayBuffer(capacity=10)
        
        # Add experiences
        for i in range(5):
            exp = Experience(
                state=torch.randn(200),
                action=i % 5,
                reward=float(i),
                next_state=torch.randn(200),
                done=False,
                log_prob=0.0,
                value=0.0,
                player_id="test"
            )
            buffer.add(exp)
        
        assert len(buffer) == 5
        
        # Sample batch
        batch = buffer.sample(3)
        assert len(batch) == 3
    
    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FullConfig()
            config.paths.log_dir = Path(tmpdir) / "logs"
            
            tokenizer = PokerTokenizer()
            model = PokerTransformer(
                state_dim=tokenizer.get_state_dim(),
                d_model=64,  # Small for testing
                n_heads=2,
                n_layers=2,
                d_ff=256,
                num_actions=5
            )
            
            logger = MetricsLogger(config.paths.log_dir, use_tensorboard=False)
            trainer = PPOTrainer(model, config, logger, device="cpu")
            
            assert trainer is not None
            assert trainer.global_step == 0
    
    def test_checkpoint_save_load(self):
        """Test checkpoint save and load."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = FullConfig()
            config.paths.log_dir = Path(tmpdir) / "logs"
            config.paths.checkpoint_dir = Path(tmpdir) / "checkpoints"
            config.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            tokenizer = PokerTokenizer()
            model = PokerTransformer(
                state_dim=tokenizer.get_state_dim(),
                d_model=64,
                n_heads=2,
                n_layers=2,
                d_ff=256,
                num_actions=5
            )
            
            logger = MetricsLogger(config.paths.log_dir, use_tensorboard=False)
            trainer = PPOTrainer(model, config, logger, device="cpu")
            
            # Save checkpoint
            checkpoint_path = config.paths.checkpoint_dir / "test.pt"
            trainer.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()
            
            # Load checkpoint
            trainer.global_step = 0  # Reset
            trainer.load_checkpoint(checkpoint_path)
            assert trainer.global_step == 0  # Should be restored


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

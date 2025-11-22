"""
Quick test of the training pipeline.

This script tests the complete training infrastructure with a minimal example.
"""

import torch
from training.model import PokerTransformer, PokerTokenizer
from training.trainer import PPOTrainer
from training.data_collector import DataCollector, ReplayBuffer, Experience
from training.metrics import MetricsLogger
from training.training_config import get_default_config

def test_training_pipeline():
    """Test the complete training pipeline."""
    print("Testing Training Pipeline")
    print("=" * 60)
    
    # Setup
    config = get_default_config()
    config.training.batch_size = 32
    config.training.device = "cpu"
    
    tokenizer = PokerTokenizer()
    state_dim = tokenizer.get_state_dim()
    
    model = PokerTransformer(
        state_dim=state_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_actions=5
    )
    
    print(f"Model created with {model.count_parameters():,} parameters")
    
    # Create components
    logger = MetricsLogger(config.paths.log_dir, use_tensorboard=False)
    trainer = PPOTrainer(model, config, logger, device="cpu")
    replay_buffer = ReplayBuffer(capacity=1000)
    data_collector = DataCollector(tokenizer, replay_buffer, gamma=0.99)
    
    # Generate some dummy experiences
    print("\nGenerating dummy training data...")
    num_experiences = 100
    
    for i in range(num_experiences):
        # Create dummy experience
        state = torch.randn(state_dim)
        action = torch.randint(0, 5, (1,)).item()
        reward = torch.randn(1).item()
        next_state = torch.randn(state_dim)
        done = i % 10 == 9  # Every 10th experience is terminal
        
        # Get model predictions
        with torch.no_grad():
            action_logits, value = model(state.unsqueeze(0))
            probs = torch.softmax(action_logits, dim=-1)
            log_prob = torch.log(probs[0, action]).item()
            value = value.item()
        
        # Add to buffer
        exp = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state if not done else None,
            done=done,
            log_prob=log_prob,
            value=value,
            player_id=f"player_{i % 6}"
        )
        replay_buffer.add(exp)
    
    print(f"Added {len(replay_buffer)} experiences to replay buffer")
    
    # Perform training update
    print("\nPerforming training update...")
    metrics = trainer.train_step(data_collector)
    
    if metrics:
        print("\nTraining Metrics:")
        print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
        print(f"  Value Loss: {metrics['value_loss']:.4f}")
        print(f"  Entropy: {metrics['entropy']:.4f}")
        print(f"  Learning Rate: {metrics['lr']:.6f}")
    else:
        print("No training update performed (not enough data)")
    
    # Test checkpoint save/load
    print("\nTesting checkpoint save/load...")
    checkpoint_path = config.paths.checkpoint_dir / "test_checkpoint.pt"
    trainer.save_checkpoint(checkpoint_path, test_data="test_value")
    print(f"Checkpoint saved to {checkpoint_path}")
    
    # Create new trainer and load
    new_model = PokerTransformer(
        state_dim=state_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_actions=5
    )
    new_trainer = PPOTrainer(new_model, config, logger, device="cpu")
    checkpoint_data = new_trainer.load_checkpoint(checkpoint_path)
    print(f"Checkpoint loaded successfully")
    print(f"Checkpoint data: {checkpoint_data}")
    
    # Cleanup
    checkpoint_path.unlink()
    logger.close()
    
    print("\n" + "=" * 60)
    print("✓ Training pipeline test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_training_pipeline()

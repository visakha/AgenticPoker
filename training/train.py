"""
Main training script for LLM poker agents.

Usage:
    python training/train.py --num-games 1000000 --parallel-games 8
    python training/train.py --resume --checkpoint path/to/checkpoint.pt
"""

import argparse
from pathlib import Path
import torch
import signal
import signal
import sys
from functools import partial

from training.training_config import FullConfig, get_default_config
from training.model import PokerTransformer, PokerTokenizer
from training.trainer import PPOTrainer
from training.metrics import MetricsLogger
from training.data_collector import DataCollector, ReplayBuffer
from training.game_simulator import GameSimulator
from agents.llm_agent import LLMAgent
from engine.domain import PlayerID


class TrainingOrchestrator:
    """Orchestrates the entire training process."""
    
    def __init__(self, config: FullConfig, resume_from: Path = None):
        """
        Initialize training orchestrator.
        
        Args:
            config: Full training configuration
            resume_from: Optional checkpoint path to resume from
        """
        self.config = config
        self.device = config.training.device
        
        # Initialize components
        print("Initializing training components...")
        
        # Tokenizer
        self.tokenizer = PokerTokenizer()
        
        # Model
        self.model = PokerTransformer(
            state_dim=self.tokenizer.get_state_dim(),
            d_model=config.model.d_model,
            n_heads=config.model.n_heads,
            n_layers=config.model.n_layers,
            d_ff=config.model.d_ff,
            num_actions=config.model.num_actions,
            dropout=config.model.dropout
        )
        print(f"Model created with {self.model.count_parameters():,} parameters")
        
        # Logger
        self.logger = MetricsLogger(config.paths.log_dir, use_tensorboard=True)
        
        # Trainer
        self.trainer = PPOTrainer(self.model, config, self.logger, self.device)
        
        # Data collector
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.data_collector = DataCollector(
            self.tokenizer,
            self.replay_buffer,
            gamma=config.training.gamma
        )
        
        # Game simulator
        self.simulator = GameSimulator(
            num_players=config.simulation.num_players,
            starting_chips=config.simulation.starting_chips,
            small_blind=config.simulation.small_blind,
            big_blind=config.simulation.big_blind,
            max_hands=config.simulation.max_hands,
            use_multiprocessing=config.simulation.use_multiprocessing,
            num_workers=config.simulation.num_workers
        )
        
        # Resume from checkpoint if provided
        if resume_from and resume_from.exists():
            self.trainer.load_checkpoint(resume_from)
        
        # Setup graceful shutdown
        self.should_stop = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nReceived interrupt signal. Saving checkpoint and exiting...")
        self.should_stop = True
    
    
    def train(self):
        """Run the main training loop."""
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        print(f"Total games: {self.config.training.num_games:,}")
        print(f"Parallel games: {self.config.training.parallel_games}")
        print(f"Games per update: {self.config.training.games_per_update}")
        print(f"Device: {self.device}")
        print("=" * 60 + "\n")
        
        games_played = 0
        updates = 0
        
        while games_played < self.config.training.num_games and not self.should_stop:
            # Collect games
            print(f"\nCollecting {self.config.training.games_per_update} games...")
            
            # Set model to eval mode for data collection
            self.model.eval()
            
            # Create agent factory with bound args
            agent_factory = partial(
                create_agent_fn,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device
            )
            
            # Simulate games
            results, stats = self.simulator.simulate_games(
                num_games=self.config.training.games_per_update,
                agent_factory=agent_factory,
                show_progress=True
            )
            
            games_played += len(results)
            
            # Log episode metrics and add to replay buffer
            for result in results:
                self.logger.log_episode(
                    episode_num=self.trainer.total_episodes,
                    reward=sum(result.final_chips.values()) / len(result.final_chips),
                    length=result.num_hands
                )
                self.trainer.total_episodes += 1
                
                # Add episodes to replay buffer
                if result.episodes:
                    for episode in result.episodes:
                        self.replay_buffer.add_episode(episode)
            
            print(f"Games played: {games_played}/{self.config.training.num_games}")
            print(f"Avg hands/game: {stats.avg_hands_per_game:.1f}")
            print(f"Games/second: {stats.games_per_second:.2f}")
            
            # Training update
            if len(self.replay_buffer) >= self.config.training.batch_size:
                print("\nPerforming training update...")
                
                # Set model to train mode
                self.model.train()
                
                # Perform update
                metrics = self.trainer.train_step(self.data_collector)
                
                if metrics:
                    updates += 1
                    print(f"Update {updates}:")
                    print(f"  Policy Loss: {metrics['policy_loss']:.4f}")
                    print(f"  Value Loss: {metrics['value_loss']:.4f}")
                    print(f"  Entropy: {metrics['entropy']:.4f}")
                    print(f"  LR: {metrics['lr']:.6f}")
            
            # Evaluation
            if updates > 0 and updates % self.config.training.eval_frequency == 0:
                print("\nRunning evaluation...")
                self.evaluate()
            
            # Checkpoint
            if updates > 0 and updates % self.config.training.checkpoint_frequency == 0:
                checkpoint_path = (
                    self.config.paths.checkpoint_dir / 
                    f"checkpoint_step_{self.trainer.global_step}.pt"
                )
                self.trainer.save_checkpoint(
                    checkpoint_path,
                    games_played=games_played,
                    updates=updates
                )
                
                # Clean up old checkpoints
                self._cleanup_old_checkpoints()
            
            # Print summary
            if updates > 0 and updates % self.config.training.log_frequency == 0:
                self.logger.print_summary()
        
        # Final checkpoint
        print("\nTraining complete! Saving final checkpoint...")
        final_path = self.config.paths.checkpoint_dir / "final_model.pt"
        self.trainer.save_checkpoint(
            final_path,
            games_played=games_played,
            updates=updates
        )
        
        # Close logger
        self.logger.close()
        
        print("\n" + "=" * 60)
        print("Training Finished!")
        print(f"Total games: {games_played:,}")
        print(f"Total updates: {updates}")
        print(f"Final checkpoint: {final_path}")
        print("=" * 60)
    
    def evaluate(self):
        """Run evaluation."""
        self.model.eval()
        
        # Create agent factory with bound args
        agent_factory = partial(
            create_agent_fn,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )
        
        # Run evaluation games
        results, stats = self.simulator.simulate_games(
            num_games=self.config.training.eval_games,
            agent_factory=agent_factory,
            show_progress=False
        )
        
        # Calculate metrics
        avg_reward = sum(
            sum(r.final_chips.values()) / len(r.final_chips)
            for r in results
        ) / len(results)
        
        win_rate = len([r for r in results if r.winner_id]) / len(results)
        
        # Log evaluation
        self.logger.log_evaluation(
            step=self.trainer.global_step,
            win_rate=win_rate,
            avg_reward=avg_reward,
            avg_hands=stats.avg_hands_per_game
        )
        
        print(f"Evaluation Results:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Hands: {stats.avg_hands_per_game:.1f}")
    
    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints."""
        checkpoints = sorted(
            self.config.paths.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        # Remove old checkpoints
        for checkpoint in checkpoints[self.config.training.keep_checkpoints:]:
            checkpoint.unlink()
            print(f"Removed old checkpoint: {checkpoint.name}")


def create_agent_fn(
    player_id: PlayerID,
    model: PokerTransformer,
    tokenizer: PokerTokenizer,
    device: str
) -> LLMAgent:
    """Standalone agent factory function."""
    return LLMAgent(
        player_id=player_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        temperature=1.0,
        deterministic=False
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train LLM poker agents")
    
    # Training arguments
    parser.add_argument(
        "--num-games",
        type=int,
        default=1_000_000,
        help="Total number of games to play"
    )
    parser.add_argument(
        "--parallel-games",
        type=int,
        default=8,
        help="Number of parallel games"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    
    # Model arguments
    parser.add_argument(
        "--d-model",
        type=int,
        default=256,
        help="Model dimension"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of transformer layers"
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file"
    )
    
    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = FullConfig.load(Path(args.config))
    else:
        config = get_default_config()
    
    # Override config with command line arguments
    config.training.num_games = args.num_games
    config.training.parallel_games = args.parallel_games
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.device = args.device
    config.model.d_model = args.d_model
    config.model.n_layers = args.n_layers
    
    # Determine checkpoint path
    resume_from = None
    if args.resume:
        if args.checkpoint:
            resume_from = Path(args.checkpoint)
        else:
            # Find latest checkpoint
            checkpoints = sorted(
                config.paths.checkpoint_dir.glob("checkpoint_step_*.pt"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if checkpoints:
                resume_from = checkpoints[0]
                print(f"Resuming from latest checkpoint: {resume_from}")
    
    # Create orchestrator and train
    orchestrator = TrainingOrchestrator(config, resume_from)
    orchestrator.train()


if __name__ == "__main__":
    main()

"""
Model evaluation framework.

Evaluates trained models by:
- Playing against baseline agents
- Calculating win rates and performance metrics
- ELO rating system
"""

# Allow running as both module and script
import sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch

from training.model import PokerTransformer, PokerTokenizer
from training.game_simulator import GameSimulator, GameResult
from agents.llm_agent import LLMAgent
from agents.functional_agent import FunctionalAgent, Personality
from engine.domain import PlayerID


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    
    model_name: str
    total_games: int
    wins: int
    losses: int
    win_rate: float
    avg_chips: float
    avg_hands_per_game: float
    vpip: float = 0.0
    pfr: float = 0.0
    aggression: float = 0.0


class ModelEvaluator:
    """Evaluates trained poker models."""
    
    def __init__(
        self,
        model: PokerTransformer,
        tokenizer: PokerTokenizer,
        device: str = "cpu"
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model to evaluate
            tokenizer: Poker tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Set model to eval mode
        self.model.eval()
    
    def evaluate_against_baselines(
        self,
        num_games: int = 100,
        num_players: int = 6
    ) -> EvaluationResult:
        """
        Evaluate model against baseline personality agents.
        
        Args:
            num_games: Number of games to play
            num_players: Number of players per game
        
        Returns:
            Evaluation results
        """
        # Create baseline personalities
        baseline_personalities = [
            Personality(
                name="Tight",
                vpip=0.2,
                pfr=0.15,
                aggression_factor=0.3,
                bluff_frequency=0.05,
                aggression=0.3
            ),
            Personality(
                name="Loose",
                vpip=0.5,
                pfr=0.3,
                aggression_factor=0.6,
                bluff_frequency=0.2,
                aggression=0.6
            ),
            Personality(
                name="Aggressive",
                vpip=0.35,
                pfr=0.25,
                aggression_factor=0.8,
                bluff_frequency=0.15,
                aggression=0.8
            ),
        ]
        
        # Create agent factory
        def agent_factory(player_id: PlayerID):
            # First player is LLM agent
            if "player_0" in str(player_id):
                return LLMAgent(
                    player_id=player_id,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    deterministic=True  # Use best action for evaluation
                )
            else:
                # Other players are baseline agents
                idx = int(str(player_id).split("_")[-1]) % len(baseline_personalities)
                personality = baseline_personalities[idx]
                return FunctionalAgent(player_id, personality)
        
        # Run simulation
        simulator = GameSimulator(
            num_players=num_players,
            starting_chips=1000,
            use_multiprocessing=False  # Sequential for evaluation
        )
        
        results, stats = simulator.simulate_games(
            num_games=num_games,
            agent_factory=agent_factory,
            show_progress=True
        )
        
        # Calculate metrics
        llm_wins = sum(1 for r in results if "player_0" in r.winner_id)
        llm_chips = []
        
        for r in results:
            for pid, chips in r.final_chips.items():
                if "player_0" in pid:
                    llm_chips.append(chips)
        
        avg_chips = sum(llm_chips) / len(llm_chips) if llm_chips else 0
        
        return EvaluationResult(
            model_name="LLM Agent",
            total_games=num_games,
            wins=llm_wins,
            losses=num_games - llm_wins,
            win_rate=llm_wins / num_games,
            avg_chips=avg_chips,
            avg_hands_per_game=stats.avg_hands_per_game
        )
    
    def compare_models(
        self,
        other_model: PokerTransformer,
        num_games: int = 100
    ) -> Tuple[float, float]:
        """
        Compare this model against another model.
        
        Args:
            other_model: Other model to compare against
            num_games: Number of games to play
        
        Returns:
            Tuple of (this_model_win_rate, other_model_win_rate)
        """
        # Create agent factory for head-to-head
        def agent_factory(player_id: PlayerID):
            if "player_0" in str(player_id):
                return LLMAgent(
                    player_id=player_id,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    deterministic=True
                )
            else:
                return LLMAgent(
                    player_id=player_id,
                    model=other_model,
                    tokenizer=self.tokenizer,
                    device=self.device,
                    deterministic=True
                )
        
        # Run simulation
        simulator = GameSimulator(num_players=2, use_multiprocessing=False)
        results, _ = simulator.simulate_games(
            num_games=num_games,
            agent_factory=agent_factory,
            show_progress=True
        )
        
        # Calculate win rates
        model1_wins = sum(1 for r in results if "player_0" in r.winner_id)
        model2_wins = num_games - model1_wins
        
        return model1_wins / num_games, model2_wins / num_games


def load_and_evaluate(
    checkpoint_path: str,
    num_games: int = 100,
    device: str = "cpu"
) -> EvaluationResult:
    """
    Load a model checkpoint and evaluate it.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_games: Number of evaluation games
        device: Device to run on
    
    Returns:
        Evaluation results
    """
    from training.training_config import ModelConfig
    
    # Load model
    tokenizer = PokerTokenizer()
    config = ModelConfig()
    
    model = PokerTransformer(
        state_dim=tokenizer.get_state_dim(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        num_actions=config.num_actions
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    evaluator = ModelEvaluator(model, tokenizer, device)
    results = evaluator.evaluate_against_baselines(num_games=num_games)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate trained poker model")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    print(f"Evaluating model from {args.checkpoint}...")
    results = load_and_evaluate(args.checkpoint, args.num_games, args.device)
    
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Model: {results.model_name}")
    print(f"Games: {results.total_games}")
    print(f"Wins: {results.wins}")
    print(f"Losses: {results.losses}")
    print(f"Win Rate: {results.win_rate:.2%}")
    print(f"Avg Chips: {results.avg_chips:.1f}")
    print(f"Avg Hands/Game: {results.avg_hands_per_game:.1f}")
    print("=" * 50)

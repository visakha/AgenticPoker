"""
High-performance game simulator for parallel poker game execution.

Handles:
- Parallel game execution using multiprocessing
- Batch processing for efficiency
- Progress tracking and statistics
- Integration with existing engine logic
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from tqdm import tqdm
import random

from engine.domain import (
    GameState, Player, PlayerID, ChipCount, PlayerStatus, Pot, HandPhase
)
from engine.logic import create_deck, shuffle_deck, draw_cards, apply_action
from agents.functional_agent import FunctionalAgent, Personality


@dataclass
class GameResult:
    """Result of a single game."""
    
    game_id: int
    winner_id: Optional[str]
    final_chips: Dict[str, int]
    num_hands: int
    total_pot: int
    duration_seconds: float


@dataclass
class SimulationStats:
    """Statistics from simulation run."""
    
    total_games: int
    total_hands: int
    total_duration: float
    games_per_second: float
    avg_hands_per_game: float
    winner_distribution: Dict[str, int]


class GameSimulator:
    """
    Simulates poker games for training data collection.
    
    Supports parallel execution for high throughput.
    """
    
    def __init__(
        self,
        num_players: int = 6,
        starting_chips: int = 1000,
        small_blind: int = 5,
        big_blind: int = 10,
        max_hands: int = 100,
        use_multiprocessing: bool = True,
        num_workers: Optional[int] = None
    ):
        """
        Initialize game simulator.
        
        Args:
            num_players: Number of players per game
            starting_chips: Starting chips for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
            max_hands: Maximum hands per game
            use_multiprocessing: Whether to use parallel execution
            num_workers: Number of worker processes (None = auto)
        """
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_hands = max_hands
        self.use_multiprocessing = use_multiprocessing
        self.num_workers = num_workers or mp.cpu_count()
    
    def simulate_games(
        self,
        num_games: int,
        agent_factory: Callable[[PlayerID], any],
        show_progress: bool = True
    ) -> Tuple[List[GameResult], SimulationStats]:
        """
        Simulate multiple games.
        
        Args:
            num_games: Number of games to simulate
            agent_factory: Function to create agents given player ID
            show_progress: Whether to show progress bar
        
        Returns:
            Tuple of (results, statistics)
        """
        start_time = time.time()
        results = []
        
        if self.use_multiprocessing and num_games > 1:
            # Parallel execution
            results = self._simulate_parallel(
                num_games, agent_factory, show_progress
            )
        else:
            # Sequential execution
            results = self._simulate_sequential(
                num_games, agent_factory, show_progress
            )
        
        # Calculate statistics
        duration = time.time() - start_time
        stats = self._calculate_stats(results, duration)
        
        return results, stats
    
    def _simulate_sequential(
        self,
        num_games: int,
        agent_factory: Callable,
        show_progress: bool
    ) -> List[GameResult]:
        """Simulate games sequentially."""
        results = []
        iterator = range(num_games)
        
        if show_progress:
            iterator = tqdm(iterator, desc="Simulating games")
        
        for game_id in iterator:
            result = self._run_single_game(game_id, agent_factory)
            results.append(result)
        
        return results
    
    def _simulate_parallel(
        self,
        num_games: int,
        agent_factory: Callable,
        show_progress: bool
    ) -> List[GameResult]:
        """Simulate games in parallel."""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all games
            futures = {
                executor.submit(self._run_single_game, game_id, agent_factory): game_id
                for game_id in range(num_games)
            }
            
            # Collect results with progress bar
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=num_games, desc="Simulating games")
            
            for future in iterator:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Game failed with error: {e}")
        
        return results
    
    def _run_single_game(
        self,
        game_id: int,
        agent_factory: Callable
    ) -> GameResult:
        """
        Run a single game simulation.
        
        Args:
            game_id: Unique game identifier
            agent_factory: Function to create agents
        
        Returns:
            Game result
        """
        start_time = time.time()
        
        # Create players
        players = []
        agents = []
        for i in range(self.num_players):
            player_id = PlayerID(f"player_{game_id}_{i}")
            player = Player(
                id=player_id,
                name=f"Player {i}",
                chips=ChipCount(self.starting_chips),
                status=PlayerStatus.ACTIVE
            )
            players.append(player)
            agents.append(agent_factory(player_id))
        
        # Initialize game state
        state = GameState(
            players=players,
            pot=Pot(ChipCount(0), [p.id for p in players]),
            dealer_index=0,
            current_player_index=0,
            deck_seed=random.randint(0, 1000000)
        )
        
        # Play hands until game ends
        num_hands = 0
        total_pot = 0
        
        while num_hands < self.max_hands:
            # Check if only one player has chips
            active_players = [p for p in state.players if p.chips > 0]
            if len(active_players) <= 1:
                break
            
            # Play one hand
            state, hand_pot = self._play_hand(state, agents)
            num_hands += 1
            total_pot += hand_pot
        
        # Determine winner (player with most chips)
        winner = max(state.players, key=lambda p: p.chips)
        
        duration = time.time() - start_time
        
        return GameResult(
            game_id=game_id,
            winner_id=str(winner.id),
            final_chips={str(p.id): p.chips for p in state.players},
            num_hands=num_hands,
            total_pot=total_pot,
            duration_seconds=duration
        )
    
    def _play_hand(
        self,
        state: GameState,
        agents: List
    ) -> Tuple[GameState, int]:
        """
        Play a single hand of poker.
        
        Args:
            state: Current game state
            agents: List of agents
        
        Returns:
            Tuple of (new_state, pot_size)
        """
        # This is a simplified version - in production, integrate with full engine
        # For now, just simulate a basic hand
        
        # Deal cards (simplified)
        deck = create_deck()
        deck = shuffle_deck(deck, state.deck_seed)
        
        # Reset for new hand
        new_players = []
        for p in state.players:
            if p.chips > 0:
                # Deal hole cards
                hole_cards, deck = draw_cards(deck, 2)
                new_players.append(Player(
                    id=p.id,
                    name=p.name,
                    chips=p.chips,
                    status=PlayerStatus.ACTIVE,
                    hole_cards=hole_cards,
                    current_bet=ChipCount(0)
                ))
            else:
                new_players.append(p)
        
        state = GameState(
            players=new_players,
            pot=Pot(ChipCount(0), [p.id for p in new_players if p.chips > 0]),
            phase=HandPhase.PREFLOP,
            community_cards=[],
            current_bet=ChipCount(self.big_blind),
            dealer_index=(state.dealer_index + 1) % len(state.players),
            deck_seed=state.deck_seed + 1
        )
        
        # Post blinds (simplified)
        # ... (full implementation would handle betting rounds, etc.)
        
        # For simulation purposes, randomly determine winner
        active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
        if active_players:
            winner = random.choice(active_players)
            pot_size = self.small_blind + self.big_blind
            
            # Award pot to winner
            new_players = []
            for p in state.players:
                if p.id == winner.id:
                    new_players.append(Player(
                        id=p.id,
                        name=p.name,
                        chips=ChipCount(p.chips + pot_size),
                        status=p.status,
                        hole_cards=[],
                        current_bet=ChipCount(0)
                    ))
                else:
                    # Deduct blind if applicable
                    blind_cost = self.small_blind if p.id != winner.id else 0
                    new_players.append(Player(
                        id=p.id,
                        name=p.name,
                        chips=ChipCount(max(0, p.chips - blind_cost)),
                        status=p.status,
                        hole_cards=[],
                        current_bet=ChipCount(0)
                    ))
            
            state = GameState(
                players=new_players,
                pot=Pot(ChipCount(0), []),
                dealer_index=state.dealer_index
            )
            
            return state, pot_size
        
        return state, 0
    
    def _calculate_stats(
        self,
        results: List[GameResult],
        duration: float
    ) -> SimulationStats:
        """Calculate statistics from results."""
        total_games = len(results)
        total_hands = sum(r.num_hands for r in results)
        
        winner_dist = {}
        for r in results:
            if r.winner_id:
                winner_dist[r.winner_id] = winner_dist.get(r.winner_id, 0) + 1
        
        return SimulationStats(
            total_games=total_games,
            total_hands=total_hands,
            total_duration=duration,
            games_per_second=total_games / duration if duration > 0 else 0,
            avg_hands_per_game=total_hands / total_games if total_games > 0 else 0,
            winner_distribution=winner_dist
        )


def simple_agent_factory(player_id: PlayerID) -> FunctionalAgent:
    """Simple agent factory for testing."""
    personality = Personality(
        name="Test",
        vpip=0.3,
        pfr=0.2,
        aggression_factor=0.5,
        bluff_frequency=0.1,
        aggression=0.5
    )
    return FunctionalAgent(player_id, personality)


if __name__ == "__main__":
    # Test simulator
    simulator = GameSimulator(
        num_players=6,
        starting_chips=1000,
        use_multiprocessing=False  # For testing
    )
    
    print("Running test simulation...")
    results, stats = simulator.simulate_games(
        num_games=10,
        agent_factory=simple_agent_factory,
        show_progress=True
    )
    
    print(f"\nSimulation Statistics:")
    print(f"Total games: {stats.total_games}")
    print(f"Total hands: {stats.total_hands}")
    print(f"Duration: {stats.total_duration:.2f}s")
    print(f"Games/second: {stats.games_per_second:.2f}")
    print(f"Avg hands/game: {stats.avg_hands_per_game:.2f}")

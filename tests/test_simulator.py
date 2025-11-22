"""Tests for game simulator."""

import pytest
from training.game_simulator import GameSimulator, simple_agent_factory


class TestGameSimulator:
    """Test game simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator can be initialized."""
        simulator = GameSimulator(
            num_players=6,
            starting_chips=1000,
            use_multiprocessing=False
        )
        assert simulator is not None
        assert simulator.num_players == 6
        assert simulator.starting_chips == 1000
    
    def test_single_game_simulation(self):
        """Test running a single game."""
        simulator = GameSimulator(
            num_players=4,
            starting_chips=1000,
            max_hands=10,
            use_multiprocessing=False
        )
        
        results, stats = simulator.simulate_games(
            num_games=1,
            agent_factory=simple_agent_factory,
            show_progress=False
        )
        
        assert len(results) == 1
        assert results[0].num_hands > 0
        assert stats.total_games == 1
    
    def test_multiple_games_simulation(self):
        """Test running multiple games."""
        simulator = GameSimulator(
            num_players=4,
            starting_chips=1000,
            max_hands=5,
            use_multiprocessing=False
        )
        
        results, stats = simulator.simulate_games(
            num_games=5,
            agent_factory=simple_agent_factory,
            show_progress=False
        )
        
        assert len(results) == 5
        assert stats.total_games == 5
        assert stats.games_per_second > 0
    
    def test_simulation_statistics(self):
        """Test simulation statistics are calculated correctly."""
        simulator = GameSimulator(
            num_players=3,
            starting_chips=1000,
            max_hands=10,
            use_multiprocessing=False
        )
        
        results, stats = simulator.simulate_games(
            num_games=3,
            agent_factory=simple_agent_factory,
            show_progress=False
        )
        
        assert stats.total_games == 3
        assert stats.total_hands > 0
        assert stats.avg_hands_per_game > 0
        assert len(stats.winner_distribution) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Unit tests for the game theory module."""

import pytest
from engine.game_theory import (
    calculate_optimal_bluff_frequency,
    calculate_minimum_defense_frequency,
    calculate_optimal_bet_size,
    calculate_fold_equity,
    get_gto_action_frequencies,
    calculate_nash_equilibrium_push_fold,
    calculate_exploitative_adjustment,
)


def test_optimal_bluff_frequency():
    """Test optimal bluffing frequency calculation."""
    freq = calculate_optimal_bluff_frequency(100, 50)
    assert freq == pytest.approx(0.333, abs=0.01)  # 1/3 bluff frequency


def test_minimum_defense_frequency():
    """Test MDF calculation."""
    mdf = calculate_minimum_defense_frequency(100, 50)
    assert mdf == pytest.approx(0.667, abs=0.01)  # 2/3 defense frequency


def test_optimal_bet_size_strong_hand():
    """Test bet sizing with strong hand."""
    bet = calculate_optimal_bet_size(100, 0.9, polarization=0.7)
    assert bet > 50  # Strong hand should bet larger
    assert bet <= 150  # Capped at 150% pot


def test_optimal_bet_size_weak_hand():
    """Test bet sizing with weak hand."""
    bet = calculate_optimal_bet_size(100, 0.2, polarization=0.3)
    assert bet < 70  # Weak hand should bet smaller


def test_fold_equity_calculation():
    """Test fold equity calculation."""
    fe = calculate_fold_equity(0.5, 100, 50)
    assert fe == 75.0  # 50% fold * 150 pot


def test_gto_frequencies_early_position_strong():
    """Test GTO frequencies for strong hand in early position."""
    freqs = get_gto_action_frequencies("early", 0.9)
    assert freqs["raise"] > freqs["call"]
    assert freqs["fold"] == 0.0


def test_gto_frequencies_late_position_marginal():
    """Test GTO frequencies for marginal hand in late position."""
    freqs = get_gto_action_frequencies("late", 0.4)
    assert freqs["fold"] < 0.5  # Can play wider in late position
    assert sum(freqs.values()) == pytest.approx(1.0, abs=0.01)


def test_nash_push_fold_short_stack():
    """Test Nash push/fold for short stack."""
    threshold = calculate_nash_equilibrium_push_fold(500, 50, 2)
    assert 0.0 < threshold < 0.5  # Short stack pushes wide


def test_nash_push_fold_deep_stack():
    """Test Nash push/fold for deep stack."""
    threshold = calculate_nash_equilibrium_push_fold(2000, 50, 2)
    assert threshold > 0.7  # Deep stack pushes tight


def test_exploitative_adjustment_loose_passive():
    """Test exploitative adjustments vs loose passive opponent."""
    adj = calculate_exploitative_adjustment(0.6, 0.2, 1.0)
    assert adj["value_bet_thinner"] is True
    assert adj["bluff_more"] is False


def test_exploitative_adjustment_tight_aggressive():
    """Test exploitative adjustments vs tight aggressive opponent."""
    adj = calculate_exploitative_adjustment(0.2, 0.18, 2.5)
    assert adj["fold_more"] is True


def test_exploitative_adjustment_maniac():
    """Test exploitative adjustments vs maniac (loose aggressive)."""
    adj = calculate_exploitative_adjustment(0.7, 0.5, 3.0)
    assert adj["call_down_lighter"] is True

"""Unit tests for the probability module."""

import pytest
from engine.domain import Card, Rank, Suit
from engine.probability import (
    calculate_win_probability,
    calculate_pot_odds,
    calculate_expected_value,
    get_hand_strength_category,
    calculate_outs,
)
from engine.hand_evaluator import HandRank


def make_card(card_str: str) -> Card:
    """Helper to create cards from strings like 'Ah' or '2c'."""
    rank_map = {
        "2": Rank.TWO,
        "3": Rank.THREE,
        "4": Rank.FOUR,
        "5": Rank.FIVE,
        "6": Rank.SIX,
        "7": Rank.SEVEN,
        "8": Rank.EIGHT,
        "9": Rank.NINE,
        "T": Rank.TEN,
        "J": Rank.JACK,
        "Q": Rank.QUEEN,
        "K": Rank.KING,
        "A": Rank.ACE,
    }
    suit_map = {"h": Suit.HEARTS, "d": Suit.DIAMONDS, "c": Suit.CLUBS, "s": Suit.SPADES}
    return Card(rank_map[card_str[0]], suit_map[card_str[1]])


def test_win_probability_pocket_aces():
    """Test that pocket aces have high win probability."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = []
    prob = calculate_win_probability(hole, community, num_opponents=1, num_simulations=100)
    assert prob > 0.7  # Pocket aces should win >70% vs random hand


def test_win_probability_with_made_hand():
    """Test win probability with a made flush."""
    hole = [make_card("Ah"), make_card("Kh")]
    community = [make_card("Qh"), make_card("Jh"), make_card("2h")]
    prob = calculate_win_probability(hole, community, num_opponents=1, num_simulations=100)
    assert prob > 0.8  # Flush should have very high equity


def test_pot_odds_calculation():
    """Test pot odds calculation."""
    odds = calculate_pot_odds(100, 50)
    assert odds == pytest.approx(0.333, abs=0.01)  # 50/(100+50) = 1/3


def test_pot_odds_zero_bet():
    """Test pot odds with zero bet to call."""
    odds = calculate_pot_odds(100, 0)
    assert odds == 1.0


def test_expected_value_positive():
    """Test positive EV calculation."""
    ev = calculate_expected_value(0.6, 100, 50)
    assert ev > 0  # 60% to win 150, 40% to lose 50 = +EV


def test_expected_value_negative():
    """Test negative EV calculation."""
    ev = calculate_expected_value(0.3, 100, 50)
    assert ev < 0  # 30% to win 150, 70% to lose 50 = -EV


def test_expected_value_breakeven():
    """Test breakeven EV."""
    ev = calculate_expected_value(0.5, 100, 50)
    assert ev == pytest.approx(0, abs=1)


def test_hand_strength_premium():
    """Test premium hand categorization."""
    aces = [make_card("Ah"), make_card("Ad")]
    assert get_hand_strength_category(aces) == "Premium"

    kings = [make_card("Kh"), make_card("Kd")]
    assert get_hand_strength_category(kings) == "Premium"


def test_hand_strength_strong():
    """Test strong hand categorization."""
    tens = [make_card("Th"), make_card("Td")]
    assert get_hand_strength_category(tens) == "Strong"

    ak = [make_card("Ah"), make_card("Kd")]
    assert get_hand_strength_category(ak) == "Strong"


def test_hand_strength_weak():
    """Test weak hand categorization."""
    trash = [make_card("7h"), make_card("2d")]
    assert get_hand_strength_category(trash) == "Weak"


def test_hand_strength_suited_connector():
    """Test suited connector categorization."""
    suited = [make_card("Jh"), make_card("Th")]
    category = get_hand_strength_category(suited)
    assert category in ["Strong", "Playable"]


def test_outs_calculation():
    """Test outs calculation for flush draw."""
    hole = [make_card("Ah"), make_card("Kh")]
    community = [make_card("Qh"), make_card("Jh"), make_card("2c")]
    # Has 4 hearts, needs 1 more for flush
    outs = calculate_outs(hole, community, HandRank.FLUSH)
    assert outs > 0  # Should have outs to make flush

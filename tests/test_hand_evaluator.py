"""Unit tests for the hand evaluator module."""

import pytest
from engine.domain import Card, Rank, Suit
from engine.hand_evaluator import evaluate_hand, HandRank


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


def test_royal_flush():
    """Test royal flush detection."""
    hole = [make_card("Ah"), make_card("Kh")]
    community = [
        make_card("Qh"),
        make_card("Jh"),
        make_card("Th"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.ROYAL_FLUSH


def test_straight_flush():
    """Test straight flush detection."""
    hole = [make_card("9h"), make_card("8h")]
    community = [
        make_card("7h"),
        make_card("6h"),
        make_card("5h"),
        make_card("Ac"),
        make_card("Kd"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.STRAIGHT_FLUSH


def test_four_of_a_kind():
    """Test four of a kind detection."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = [
        make_card("Ac"),
        make_card("As"),
        make_card("Kh"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.FOUR_OF_A_KIND


def test_full_house():
    """Test full house detection."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = [
        make_card("Ac"),
        make_card("Kh"),
        make_card("Kd"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.FULL_HOUSE


def test_flush():
    """Test flush detection."""
    hole = [make_card("Ah"), make_card("2h")]
    community = [
        make_card("5h"),
        make_card("9h"),
        make_card("Kh"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.FLUSH


def test_straight():
    """Test straight detection."""
    hole = [make_card("9h"), make_card("8d")]
    community = [
        make_card("7c"),
        make_card("6s"),
        make_card("5h"),
        make_card("Ac"),
        make_card("Kd"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.STRAIGHT


def test_ace_low_straight():
    """Test Ace-low straight (wheel) detection."""
    hole = [make_card("Ah"), make_card("2d")]
    community = [
        make_card("3c"),
        make_card("4s"),
        make_card("5h"),
        make_card("Kc"),
        make_card("Qd"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.STRAIGHT
    assert kickers[0] == 5  # Ace-low straight has 5 as high card


def test_three_of_a_kind():
    """Test three of a kind detection."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = [
        make_card("Ac"),
        make_card("Kh"),
        make_card("Qd"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.THREE_OF_A_KIND


def test_two_pair():
    """Test two pair detection."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = [
        make_card("Kc"),
        make_card("Kh"),
        make_card("Qd"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.TWO_PAIR


def test_pair():
    """Test pair detection."""
    hole = [make_card("Ah"), make_card("Ad")]
    community = [
        make_card("Kc"),
        make_card("Qh"),
        make_card("Jd"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.PAIR


def test_high_card():
    """Test high card detection."""
    hole = [make_card("Ah"), make_card("Kd")]
    community = [
        make_card("Qc"),
        make_card("Jh"),
        make_card("9d"),
        make_card("2c"),
        make_card("3d"),
    ]
    rank, kickers = evaluate_hand(hole, community)
    assert rank == HandRank.HIGH_CARD


def test_kicker_comparison():
    """Test that kickers are properly sorted for comparison."""
    hole1 = [make_card("Ah"), make_card("Kd")]
    hole2 = [make_card("Ah"), make_card("Qd")]
    community = [
        make_card("2c"),
        make_card("3h"),
        make_card("4d"),
        make_card("5c"),
        make_card("7d"),
    ]

    rank1, kickers1 = evaluate_hand(hole1, community)
    rank2, kickers2 = evaluate_hand(hole2, community)

    assert rank1 == rank2 == HandRank.HIGH_CARD
    assert kickers1 > kickers2  # King kicker beats Queen kicker

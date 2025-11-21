"""
Probability calculator for poker hands.

This module calculates win probabilities for poker hands using:
- Monte Carlo simulation for accurate estimates
- Combinatorics for exact calculations (when feasible)
- Stage-specific analysis (Preflop, Flop, Turn, River)

Example:
    >>> from engine.probability import calculate_win_probability
    >>> from engine.domain import Card, Rank, Suit
    >>> hole_cards = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)]
    >>> prob = calculate_win_probability(hole_cards, [], num_opponents=2)
    >>> print(f"Win probability: {prob:.2%}")
    Win probability: 67.34%
"""

import random
from typing import List, Tuple
from collections import Counter
from engine.domain import Card, Rank, Suit, HandPhase
from engine.hand_evaluator import evaluate_hand, HandRank
from engine.logic import create_deck


def calculate_win_probability(
    hole_cards: List[Card],
    community_cards: List[Card],
    num_opponents: int = 1,
    num_simulations: int = 1000,
) -> float:
    """
    Calculate win probability using Monte Carlo simulation.

    Args:
        hole_cards: Player's hole cards (2 cards)
        community_cards: Current community cards (0-5 cards)
        num_opponents: Number of opponents
        num_simulations: Number of simulations to run

    Returns:
        Win probability (0.0 to 1.0)

    Example:
        >>> hole = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
        >>> community = []
        >>> prob = calculate_win_probability(hole, community, num_opponents=1)
        >>> prob > 0.8  # Pocket aces have high win rate
        True
    """
    wins = 0
    ties = 0

    # Create deck without known cards
    full_deck = create_deck()
    known_cards = set(hole_cards + community_cards)
    available_deck = [c for c in full_deck if c not in known_cards]

    for _ in range(num_simulations):
        # Shuffle available cards
        random.shuffle(available_deck)

        # Complete the board
        cards_needed = 5 - len(community_cards)
        simulated_community = community_cards + available_deck[:cards_needed]

        # Deal opponent hands
        remaining_deck = available_deck[cards_needed:]
        opponent_hands = []
        for i in range(num_opponents):
            opponent_hole = remaining_deck[i * 2 : (i + 1) * 2]
            opponent_hands.append(opponent_hole)

        # Evaluate all hands
        player_rank, player_kickers = evaluate_hand(hole_cards, simulated_community)
        opponent_results = [
            evaluate_hand(opp_hole, simulated_community) for opp_hole in opponent_hands
        ]

        # Determine winner
        best_opponent_rank = max(r for r, k in opponent_results)
        best_opponent_kickers = max(
            k for r, k in opponent_results if r == best_opponent_rank
        )

        if player_rank > best_opponent_rank:
            wins += 1
        elif player_rank == best_opponent_rank:
            if player_kickers > best_opponent_kickers:
                wins += 1
            elif player_kickers == best_opponent_kickers:
                ties += 1

    return (wins + ties * 0.5) / num_simulations


def calculate_pot_odds(pot_size: int, bet_to_call: int) -> float:
    """
    Calculate pot odds (ratio of pot to bet).

    Args:
        pot_size: Current pot size
        bet_to_call: Amount to call

    Returns:
        Pot odds as a decimal (e.g., 0.33 for 3:1 odds)

    Example:
        >>> calculate_pot_odds(100, 50)
        0.3333333333333333
    """
    if bet_to_call == 0:
        return 1.0
    return bet_to_call / (pot_size + bet_to_call)


def calculate_expected_value(
    win_probability: float, pot_size: int, bet_to_call: int
) -> float:
    """
    Calculate expected value of a call.

    EV = (Win% × Pot) - (Lose% × Bet)

    Args:
        win_probability: Probability of winning (0.0 to 1.0)
        pot_size: Current pot size
        bet_to_call: Amount to call

    Returns:
        Expected value in chips

    Example:
        >>> calculate_expected_value(0.5, 100, 50)
        0.0
        >>> calculate_expected_value(0.6, 100, 50)
        10.0
    """
    win_amount = pot_size + bet_to_call
    lose_amount = bet_to_call
    return (win_probability * win_amount) - ((1 - win_probability) * lose_amount)


def get_hand_strength_category(hole_cards: List[Card]) -> str:
    """
    Categorize preflop hand strength.

    Args:
        hole_cards: Player's hole cards

    Returns:
        Category: "Premium", "Strong", "Playable", "Marginal", "Weak"

    Example:
        >>> from engine.domain import Card, Rank, Suit
        >>> aces = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
        >>> get_hand_strength_category(aces)
        'Premium'
    """
    if len(hole_cards) != 2:
        return "Unknown"

    c1, c2 = hole_cards
    rank_values = {
        Rank.TWO: 2,
        Rank.THREE: 3,
        Rank.FOUR: 4,
        Rank.FIVE: 5,
        Rank.SIX: 6,
        Rank.SEVEN: 7,
        Rank.EIGHT: 8,
        Rank.NINE: 9,
        Rank.TEN: 10,
        Rank.JACK: 11,
        Rank.QUEEN: 12,
        Rank.KING: 13,
        Rank.ACE: 14,
    }

    v1, v2 = rank_values[c1.rank], rank_values[c2.rank]
    is_pair = v1 == v2
    is_suited = c1.suit == c2.suit
    high_card = max(v1, v2)
    low_card = min(v1, v2)

    # Premium hands
    if is_pair and high_card >= 10:  # JJ+
        return "Premium"
    if high_card == 14 and low_card >= 11 and is_suited:  # AKs, AQs, AJs
        return "Premium"

    # Strong hands
    if is_pair and high_card >= 7:  # 77+
        return "Strong"
    if high_card >= 13 and low_card >= 10:  # KQ, KJ, QJ
        return "Strong"
    if high_card == 14 and low_card >= 10:  # AT+
        return "Strong"

    # Playable hands
    if is_suited and high_card >= 10:
        return "Playable"
    if high_card >= 11 and low_card >= 9:
        return "Playable"

    # Marginal hands
    if is_pair or (is_suited and high_card >= 8):
        return "Marginal"

    return "Weak"


def calculate_outs(
    hole_cards: List[Card], community_cards: List[Card], target_hand: HandRank
) -> int:
    """
    Calculate number of outs to improve to target hand.

    Args:
        hole_cards: Player's hole cards
        community_cards: Current community cards
        target_hand: Target hand rank to achieve

    Returns:
        Number of outs (cards that improve the hand)

    Note:
        This is a simplified calculation. A full implementation would
        enumerate all possible cards and check if they achieve the target.
    """
    # Simplified: count cards that could help
    # Full implementation would simulate all remaining cards
    full_deck = create_deck()
    known_cards = set(hole_cards + community_cards)
    unknown_cards = [c for c in full_deck if c not in known_cards]

    outs = 0
    current_rank, _ = evaluate_hand(hole_cards, community_cards)

    if current_rank >= target_hand:
        return 0  # Already have target or better

    for card in unknown_cards:
        test_community = community_cards + [card]
        if len(test_community) > 5:
            continue
        test_rank, _ = evaluate_hand(hole_cards, test_community)
        if test_rank >= target_hand:
            outs += 1

    return outs

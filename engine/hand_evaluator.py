"""
Hand evaluator using the Treys library.

This module provides an adapter layer between our domain models and the Treys library,
maintaining our API while leveraging Treys' fast bit-arithmetic-based evaluation.

Treys Performance: ~3.2M evaluations/second
Our previous implementation: ~50K evaluations/second (itertools combinations)

Example:
    >>> from engine.domain import Card, Rank, Suit
    >>> from engine.hand_evaluator import evaluate_hand
    >>> hole = [Card(Rank.ACE, Suit.SPADES), Card(Rank.KING, Suit.SPADES)]
    >>> community = [Card(Rank.QUEEN, Suit.SPADES), Card(Rank.JACK, Suit.SPADES), Card(Rank.TEN, Suit.SPADES)]
    >>> rank, kickers = evaluate_hand(hole, community)
    >>> rank == HandRank.ROYAL_FLUSH
    True
"""

from enum import IntEnum
from typing import List, Tuple
from treys import Card as TreysCard, Evaluator as TreysEvaluator
from .domain import Card, Rank, Suit


class HandRank(IntEnum):
    """
    Hand rankings from lowest to highest.

    Note: Treys uses a different scale (1-7462 where 1 is best).
    We convert to our enum for consistency.
    """

    HIGH_CARD = 1
    PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


# Treys rank class thresholds (from Treys source code)
# Treys uses 1-7462 scale where 1 is Royal Flush
TREYS_RANK_CLASS_THRESHOLDS = {
    1: HandRank.ROYAL_FLUSH,  # 1 (only one royal flush)
    10: HandRank.STRAIGHT_FLUSH,  # 2-10
    166: HandRank.FOUR_OF_A_KIND,  # 11-166
    322: HandRank.FULL_HOUSE,  # 167-322
    1599: HandRank.FLUSH,  # 323-1599
    1609: HandRank.STRAIGHT,  # 1600-1609
    2467: HandRank.THREE_OF_A_KIND,  # 1610-2467
    3325: HandRank.TWO_PAIR,  # 2468-3325
    6185: HandRank.PAIR,  # 3326-6185
    7462: HandRank.HIGH_CARD,  # 6186-7462
}


def _convert_to_treys_card(card: Card) -> int:
    """
    Convert our Card to Treys card integer.

    Args:
        card: Our Card object

    Returns:
        Treys card integer
    """
    rank_map = {
        Rank.TWO: "2",
        Rank.THREE: "3",
        Rank.FOUR: "4",
        Rank.FIVE: "5",
        Rank.SIX: "6",
        Rank.SEVEN: "7",
        Rank.EIGHT: "8",
        Rank.NINE: "9",
        Rank.TEN: "T",
        Rank.JACK: "J",
        Rank.QUEEN: "Q",
        Rank.KING: "K",
        Rank.ACE: "A",
    }
    suit_map = {
        Suit.SPADES: "s",
        Suit.HEARTS: "h",
        Suit.DIAMONDS: "d",
        Suit.CLUBS: "c",
    }
    card_str = rank_map[card.rank] + suit_map[card.suit]
    return TreysCard.new(card_str)


def _treys_score_to_hand_rank(score: int) -> HandRank:
    """
    Convert Treys score (1-7462) to our HandRank enum.

    Args:
        score: Treys score (1 is best, 7462 is worst)

    Returns:
        Our HandRank enum
    """
    for threshold, rank in TREYS_RANK_CLASS_THRESHOLDS.items():
        if score <= threshold:
            return rank
    return HandRank.HIGH_CARD


def _get_kickers_from_score(score: int) -> List[int]:
    """
    Extract kicker values from Treys score.

    Note: Treys doesn't directly expose kickers, so we use the score
    as a proxy. Lower score = better hand.

    Args:
        score: Treys score

    Returns:
        List of kicker values (for comparison)
    """
    # Treys score is already a unique ranking
    # We return it as a single-element list for compatibility
    # Lower score = better hand, so negate for comparison
    return [-score]


def evaluate_hand(
    hole_cards: List[Card], community_cards: List[Card]
) -> Tuple[HandRank, List[int]]:
    """
    Evaluate the best 5-card hand from hole cards and community cards.

    Uses Treys library for fast evaluation (~3.2M hands/sec).

    Args:
        hole_cards: Player's hole cards (2 cards)
        community_cards: Community cards (0-5 cards)

    Returns:
        Tuple of (HandRank, kickers) where kickers is a list for comparison

    Example:
        >>> from engine.domain import Card, Rank, Suit
        >>> hole = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.ACE, Suit.DIAMONDS)]
        >>> community = [Card(Rank.ACE, Suit.CLUBS), Card(Rank.KING, Suit.HEARTS), Card(Rank.QUEEN, Suit.HEARTS)]
        >>> rank, kickers = evaluate_hand(hole, community)
        >>> rank == HandRank.THREE_OF_A_KIND
        True
    """
    # Treys requires 5, 6, or 7 total cards
    total_cards = len(hole_cards) + len(community_cards)
    
    if total_cards < 5:
        # Not enough cards to evaluate - return high card
        return HandRank.HIGH_CARD, [0]
    
    # Convert to Treys cards
    treys_hole = [_convert_to_treys_card(c) for c in hole_cards]
    treys_community = [_convert_to_treys_card(c) for c in community_cards]

    # Evaluate with Treys
    evaluator = TreysEvaluator()
    
    # Treys.evaluate expects (board, hand) where board is community cards
    score = evaluator.evaluate(treys_community, treys_hole)

    # Convert to our format
    hand_rank = _treys_score_to_hand_rank(score)
    kickers = _get_kickers_from_score(score)

    return hand_rank, kickers


def get_hand_description(rank: HandRank) -> str:
    """
    Get human-readable description of hand rank.

    Args:
        rank: HandRank enum

    Returns:
        String description

    Example:
        >>> get_hand_description(HandRank.ROYAL_FLUSH)
        'Royal Flush'
    """
    descriptions = {
        HandRank.HIGH_CARD: "High Card",
        HandRank.PAIR: "Pair",
        HandRank.TWO_PAIR: "Two Pair",
        HandRank.THREE_OF_A_KIND: "Three of a Kind",
        HandRank.STRAIGHT: "Straight",
        HandRank.FLUSH: "Flush",
        HandRank.FULL_HOUSE: "Full House",
        HandRank.FOUR_OF_A_KIND: "Four of a Kind",
        HandRank.STRAIGHT_FLUSH: "Straight Flush",
        HandRank.ROYAL_FLUSH: "Royal Flush",
    }
    return descriptions.get(rank, "Unknown")

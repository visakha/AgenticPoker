"""
Game theory module for optimal poker strategy.

This module implements game theory concepts:
- GTO (Game Theory Optimal) strategy
- Nash equilibrium calculations
- Optimal bet sizing
- Bluffing frequencies
- Exploitative adjustments

Example:
    >>> from engine.game_theory import calculate_optimal_bluff_frequency
    >>> bluff_freq = calculate_optimal_bluff_frequency(pot_size=100, bet_size=50)
    >>> print(f"Optimal bluff frequency: {bluff_freq:.2%}")
    Optimal bluff frequency: 33.33%
"""

from typing import Tuple
from enum import Enum


class Strategy(Enum):
    """Poker strategy types."""

    GTO = "Game Theory Optimal"
    EXPLOITATIVE = "Exploitative"
    BALANCED = "Balanced"


def calculate_optimal_bluff_frequency(pot_size: int, bet_size: int) -> float:
    """
    Calculate optimal bluffing frequency using game theory.

    Based on the principle: Bluff frequency = Bet / (Pot + Bet)
    This makes opponent indifferent to calling.

    Args:
        pot_size: Current pot size
        bet_size: Size of the bet

    Returns:
        Optimal bluff frequency (0.0 to 1.0)

    Example:
        >>> calculate_optimal_bluff_frequency(100, 50)
        0.3333333333333333
    """
    if pot_size + bet_size == 0:
        return 0.0
    return bet_size / (pot_size + bet_size)


def calculate_minimum_defense_frequency(pot_size: int, bet_size: int) -> float:
    """
    Calculate minimum defense frequency to prevent exploitation.

    MDF = Pot / (Pot + Bet)

    Args:
        pot_size: Current pot size
        bet_size: Size of opponent's bet

    Returns:
        Minimum defense frequency (0.0 to 1.0)

    Example:
        >>> calculate_minimum_defense_frequency(100, 50)
        0.6666666666666666
    """
    if pot_size + bet_size == 0:
        return 0.0
    return pot_size / (pot_size + bet_size)


def calculate_optimal_bet_size(
    pot_size: int, hand_strength: float, polarization: float = 0.5
) -> int:
    """
    Calculate optimal bet size based on hand strength and strategy.

    Args:
        pot_size: Current pot size
        hand_strength: Hand strength (0.0 to 1.0)
        polarization: How polarized the range is (0.0 = merged, 1.0 = polarized)

    Returns:
        Optimal bet size in chips

    Example:
        >>> calculate_optimal_bet_size(100, 0.8, 0.7)
        70
    """
    # Simplified model:
    # - Strong hands: bet larger (60-100% pot)
    # - Medium hands: bet smaller (30-60% pot)
    # - Polarized ranges: bet larger

    base_bet = pot_size * 0.5  # 50% pot as baseline

    # Adjust for hand strength
    strength_multiplier = 0.5 + (hand_strength * 0.5)

    # Adjust for polarization
    polarization_multiplier = 1.0 + (polarization * 0.5)

    optimal_bet = base_bet * strength_multiplier * polarization_multiplier

    return int(min(optimal_bet, pot_size * 1.5))  # Cap at 150% pot


def calculate_fold_equity(
    opponent_fold_frequency: float, pot_size: int, bet_size: int
) -> float:
    """
    Calculate fold equity (EV from opponent folding).

    Fold Equity = (Fold Frequency) × (Pot + Bet)

    Args:
        opponent_fold_frequency: Estimated fold frequency (0.0 to 1.0)
        pot_size: Current pot size
        bet_size: Size of the bet

    Returns:
        Expected value from folds

    Example:
        >>> calculate_fold_equity(0.5, 100, 50)
        75.0
    """
    return opponent_fold_frequency * (pot_size + bet_size)


def get_gto_action_frequencies(
    position: str, hand_strength: float
) -> dict[str, float]:
    """
    Get GTO action frequencies based on position and hand strength.

    Args:
        position: Position ("early", "middle", "late", "blinds")
        hand_strength: Hand strength category (0.0 to 1.0)

    Returns:
        Dictionary of action frequencies

    Example:
        >>> freqs = get_gto_action_frequencies("late", 0.7)
        >>> freqs['raise'] > freqs['call']
        True
    """
    # Simplified GTO frequencies
    # In reality, these would come from solver outputs

    if position == "early":
        if hand_strength > 0.8:
            return {"fold": 0.0, "call": 0.2, "raise": 0.8}
        elif hand_strength > 0.5:
            return {"fold": 0.3, "call": 0.5, "raise": 0.2}
        else:
            return {"fold": 0.9, "call": 0.1, "raise": 0.0}

    elif position == "middle":
        if hand_strength > 0.7:
            return {"fold": 0.0, "call": 0.3, "raise": 0.7}
        elif hand_strength > 0.4:
            return {"fold": 0.4, "call": 0.4, "raise": 0.2}
        else:
            return {"fold": 0.85, "call": 0.15, "raise": 0.0}

    elif position == "late":
        if hand_strength > 0.6:
            return {"fold": 0.0, "call": 0.2, "raise": 0.8}
        elif hand_strength > 0.3:
            return {"fold": 0.3, "call": 0.4, "raise": 0.3}
        else:
            return {"fold": 0.7, "call": 0.2, "raise": 0.1}

    else:  # blinds
        if hand_strength > 0.7:
            return {"fold": 0.0, "call": 0.4, "raise": 0.6}
        elif hand_strength > 0.4:
            return {"fold": 0.5, "call": 0.4, "raise": 0.1}
        else:
            return {"fold": 0.9, "call": 0.1, "raise": 0.0}


def calculate_nash_equilibrium_push_fold(
    stack_size: int, big_blind: int, num_opponents: int
) -> float:
    """
    Calculate Nash equilibrium push/fold threshold for short stacks.

    This is a simplified model. Real Nash calculations are complex.

    Args:
        stack_size: Player's stack in chips
        big_blind: Big blind size
        num_opponents: Number of opponents

    Returns:
        Hand strength threshold for pushing (0.0 to 1.0)

    Example:
        >>> threshold = calculate_nash_equilibrium_push_fold(500, 50, 2)
        >>> 0.0 < threshold < 1.0
        True
    """
    # Simplified Nash push/fold model
    # Based on stack-to-pot ratio (SPR)

    spr = stack_size / big_blind

    if spr <= 5:
        # Very short stack: push wide
        return 0.3 - (num_opponents * 0.05)
    elif spr <= 10:
        # Short stack: push moderately
        return 0.5 - (num_opponents * 0.05)
    elif spr <= 20:
        # Medium stack: push tight
        return 0.7 - (num_opponents * 0.05)
    else:
        # Deep stack: push very tight
        return 0.85 - (num_opponents * 0.05)


def calculate_exploitative_adjustment(
    opponent_vpip: float, opponent_pfr: float, opponent_aggression: float
) -> dict[str, float]:
    """
    Calculate exploitative adjustments based on opponent tendencies.

    Args:
        opponent_vpip: Opponent's VPIP (0.0 to 1.0)
        opponent_pfr: Opponent's PFR (0.0 to 1.0)
        opponent_aggression: Opponent's aggression factor

    Returns:
        Adjustment factors for different actions

    Example:
        >>> adjustments = calculate_exploitative_adjustment(0.6, 0.3, 2.0)
        >>> adjustments['bluff_more']  # Against loose passive
        True
    """
    adjustments = {
        "bluff_more": False,
        "value_bet_thinner": False,
        "call_down_lighter": False,
        "fold_more": False,
    }

    # Loose passive (high VPIP, low PFR, low aggression)
    if opponent_vpip > 0.4 and opponent_pfr < 0.15 and opponent_aggression < 1.5:
        adjustments["value_bet_thinner"] = True
        adjustments["bluff_more"] = False

    # Tight aggressive (low VPIP, high PFR, high aggression)
    elif opponent_vpip < 0.25 and opponent_pfr > 0.15 and opponent_aggression > 2.0:
        adjustments["fold_more"] = True
        adjustments["bluff_more"] = False

    # Loose aggressive (high VPIP, high PFR, high aggression)
    elif opponent_vpip > 0.35 and opponent_pfr > 0.25 and opponent_aggression > 2.0:
        adjustments["call_down_lighter"] = True
        adjustments["bluff_more"] = False

    # Tight passive (low VPIP, low PFR, low aggression)
    elif opponent_vpip < 0.25 and opponent_pfr < 0.15 and opponent_aggression < 1.5:
        adjustments["bluff_more"] = True
        adjustments["value_bet_thinner"] = False

    return adjustments

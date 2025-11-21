"""
Domain models for the Agentic Poker engine.

This module defines the core immutable data structures used throughout the poker game:
- Card, Rank, Suit: Representing playing cards
- Player: Individual player state with chips, cards, and status
- Pot: Chip pool and eligible players
- GameState: Complete game state snapshot

All dataclasses are frozen (immutable) to support functional programming patterns.

Example:
    >>> from engine.domain import Card, Rank, Suit
    >>> ace_of_spades = Card(Rank.ACE, Suit.SPADES)
    >>> print(ace_of_spades)
    A♠
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, NewType
from uuid import UUID, uuid4

# --- Types ---
PlayerID = NewType("PlayerID", str)
"""Unique identifier for a player."""

ChipCount = NewType("ChipCount", int)
"""Number of chips (always non-negative)."""


# --- Enums ---
class Suit(Enum):
    """Playing card suits with Unicode symbols."""

    HEARTS = "♥"
    DIAMONDS = "♦"
    CLUBS = "♣"
    SPADES = "♠"


class Rank(Enum):
    """Playing card ranks from Two to Ace."""

    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    TEN = "T"
    JACK = "J"
    QUEEN = "Q"
    KING = "K"
    ACE = "A"


class HandPhase(Enum):
    """Game phases in Texas Hold'em."""

    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


class PlayerStatus(Enum):
    """Player status during a hand."""

    ACTIVE = auto()
    FOLDED = auto()
    ALL_IN = auto()
    SITTING_OUT = auto()


class ActionType(Enum):
    """Possible player actions."""

    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE = auto()
    ALL_IN = auto()


# --- Immutable Data Structures ---


@dataclass(frozen=True)
class Card:
    """
    An immutable playing card.

    Attributes:
        rank: The card's rank (2-A)
        suit: The card's suit (♥♦♣♠)

    Example:
        >>> card = Card(Rank.KING, Suit.HEARTS)
        >>> str(card)
        'K♥'
    """

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        """Return string representation like 'A♠' or 'K♥'."""
        return f"{self.rank.value}{self.suit.value}"


@dataclass(frozen=True)
class Player:
    """
    Immutable player state snapshot.

    Attributes:
        id: Unique player identifier
        name: Display name
        chips: Current chip count
        status: Current status (ACTIVE, FOLDED, ALL_IN, SITTING_OUT)
        hole_cards: Player's private cards (0-2 cards)
        current_bet: Amount bet in current betting round
        is_human: True if human player, False if AI
        personality: Reference to personality config name
        acted_this_round: Whether player has acted in current betting round
    """

    id: PlayerID
    name: str
    chips: ChipCount
    status: PlayerStatus = PlayerStatus.ACTIVE
    hole_cards: List[Card] = field(default_factory=list)
    current_bet: ChipCount = ChipCount(0)
    is_human: bool = False
    personality: str = "Neutral"
    acted_this_round: bool = False


@dataclass(frozen=True)
class Pot:
    """
    Chip pot with eligible players.

    Attributes:
        amount: Total chips in pot
        eligible_players: List of player IDs who can win this pot
    """

    amount: ChipCount
    eligible_players: List[PlayerID]


@dataclass(frozen=True)
class GameState:
    """
    Complete immutable game state snapshot.

    This is the core data structure representing the entire game at a point in time.
    All game logic functions take a GameState and return a new GameState.

    Attributes:
        id: Unique game identifier
        phase: Current hand phase (PREFLOP, FLOP, TURN, RIVER, SHOWDOWN)
        community_cards: Shared cards on the board (0-5 cards)
        pot: Current pot
        current_bet: Amount to call in current betting round
        players: List of all players
        dealer_index: Index of dealer button
        current_player_index: Index of player whose turn it is
        deck_seed: Random seed for reproducible shuffles
        history: Log of all actions taken

    Example:
        >>> state = GameState(players=[...])
        >>> new_state = apply_action(state, player_id, ActionType.RAISE, ChipCount(100))
    """

    id: UUID = field(default_factory=uuid4)
    phase: HandPhase = HandPhase.PREFLOP
    community_cards: List[Card] = field(default_factory=list)
    pot: Pot = field(default_factory=lambda: Pot(ChipCount(0), []))
    current_bet: ChipCount = ChipCount(0)
    players: List[Player] = field(default_factory=list)
    dealer_index: int = 0
    current_player_index: int = 0
    deck_seed: int = 0
    history: List[str] = field(default_factory=list)

    def get_player(self, player_id: PlayerID) -> Optional[Player]:
        """
        Find a player by ID.

        Args:
            player_id: Player identifier to search for

        Returns:
            Player object if found, None otherwise
        """
        for p in self.players:
            if p.id == player_id:
                return p
        return None

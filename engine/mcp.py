from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal
from .domain import GameState, PlayerID, ActionType, ChipCount

# --- MCP Message Types ---


@dataclass(frozen=True)
class MCPMessage:
    sender: str
    recipient: str
    message_type: str
    payload: Any  # Can be Dict or Dataclass


# --- Specific Payloads ---


@dataclass(frozen=True)
class GameStateUpdate:
    """Sent by Dealer to Players to update their view of the world."""

    state: GameState


@dataclass(frozen=True)
class ActionRequest:
    """Sent by Dealer to a specific Player to request a move."""

    valid_actions: List[ActionType]
    min_bet: ChipCount
    max_bet: ChipCount


@dataclass(frozen=True)
class ActionResponse:
    """Sent by Player to Dealer with their chosen move."""

    action_type: ActionType
    amount: ChipCount = ChipCount(0)
    dialogue: Optional[str] = None
    reasoning: str = ""  # Agent's internal thought process


@dataclass(frozen=True)
class DialogueEvent:
    """Sent by any agent to broadcast chat/trash talk."""

    text: str
    emotion: str


# --- Protocol Helper ---


def create_update_message(dealer_id: str, player_id: str, state: GameState) -> MCPMessage:
    return MCPMessage(
        sender=dealer_id,
        recipient=player_id,
        message_type="GAME_STATE_UPDATE",
        payload={"state": state},
    )

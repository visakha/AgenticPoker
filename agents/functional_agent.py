from typing import Protocol, Dict, Any, Optional, Tuple, TypedDict
from dataclasses import dataclass
import yaml
from engine.domain import GameState, PlayerID, ActionType, ChipCount
from engine.mcp import MCPMessage, ActionResponse, ActionRequest, GameStateUpdate
from .dialogue_manager import DialogueManager

# --- Agent Protocol ---


class Agent(Protocol):
    def receive_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Pure(ish) function to process a message and optionally return a response.
        """
        ...


# --- Functional Agent ---


class Personality(TypedDict):
    name: str
    vpip: float
    pfr: float
    aggression_factor: float
    bluff_frequency: float
    aggression: float  # Alias


class FunctionalAgent:
    def __init__(self, player_id: PlayerID, personality: Personality):
        self.player_id = player_id
        self.personality = personality
        self.current_game_state: Optional[GameState] = None
        self.dialogue_manager = DialogueManager(use_llm=False)

    def receive_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        if message.message_type == "GAME_STATE_UPDATE":
            # Store state for decision making
            if isinstance(message.payload, dict) and "state" in message.payload:
                self.current_game_state = message.payload["state"]
            return None

        elif message.message_type == "ACTION_REQUEST":
            action, amount = self._decide_action()

            # Generate dialogue
            dialogue = None
            if self.current_game_state:
                # Find self in state
                me = next(
                    (p for p in self.current_game_state.players if p.id == self.player_id), None
                )
                if me:
                    dialogue = self.dialogue_manager.generate_dialogue(
                        me, action, self.current_game_state
                    )

            return MCPMessage(
                sender=self.player_id,
                recipient="Dealer",
                message_type="ACTION_RESPONSE",
                payload=ActionResponse(action_type=action, amount=amount, dialogue=dialogue),
            )

        return None

    def _decide_action(self) -> Tuple[ActionType, ChipCount]:
        """
        The 'Think' step. Uses personality, probability, and game theory to decide action.
        """
        import random
        from engine.probability import (
            calculate_win_probability,
            calculate_pot_odds,
            calculate_expected_value,
            get_hand_strength_category,
        )
        from engine.game_theory import (
            calculate_optimal_bet_size,
            get_gto_action_frequencies,
        )

        # If no game state, fall back to simple logic
        if not self.current_game_state:
            if self.personality["aggression_factor"] > 0.7:
                if random.random() < 0.5:
                    return ActionType.RAISE, ChipCount(20)
            return ActionType.CHECK, ChipCount(0)

        # Find self in state
        me = next(
            (p for p in self.current_game_state.players if p.id == self.player_id),
            None,
        )
        if not me or not me.hole_cards:
            return ActionType.CHECK, ChipCount(0)

        # Calculate win probability
        num_opponents = len(
            [
                p
                for p in self.current_game_state.players
                if p.status.name == "ACTIVE" and p.id != self.player_id
            ]
        )
        win_prob = calculate_win_probability(
            me.hole_cards,
            self.current_game_state.community_cards,
            num_opponents=max(1, num_opponents),
            num_simulations=500,  # Reduced for speed
        )

        # Calculate pot odds if there's a bet to call
        pot_size = self.current_game_state.pot.amount
        bet_to_call = self.current_game_state.current_bet - me.current_bet

        # Get hand strength category
        hand_category = get_hand_strength_category(me.hole_cards)

        # Determine position (simplified)
        position = "late" if self.current_game_state.current_player_index > 3 else "early"

        # Get GTO frequencies
        gto_freqs = get_gto_action_frequencies(position, win_prob)

        # Blend GTO with personality
        # Aggressive players raise more, tight players fold more
        aggression = self.personality["aggression_factor"]
        vpip = self.personality["vpip"]

        # Adjust frequencies based on personality
        adjusted_raise_freq = gto_freqs["raise"] * (0.5 + aggression * 0.5)
        adjusted_fold_freq = gto_freqs["fold"] * (2.0 - vpip)

        # Normalize
        total = adjusted_raise_freq + gto_freqs["call"] + adjusted_fold_freq
        if total > 0:
            adjusted_raise_freq /= total
            adjusted_fold_freq /= total

        # Make decision
        rand = random.random()

        if rand < adjusted_fold_freq and bet_to_call > 0:
            return ActionType.FOLD, ChipCount(0)
        elif rand < adjusted_fold_freq + adjusted_raise_freq:
            # Calculate optimal bet size
            optimal_bet = calculate_optimal_bet_size(
                pot_size, win_prob, polarization=aggression
            )
            raise_amount = max(
                self.current_game_state.current_bet + optimal_bet, me.current_bet + 20
            )
            return ActionType.RAISE, ChipCount(raise_amount)
        else:
            # Call or check
            if bet_to_call > 0:
                # Check if call is +EV
                ev = calculate_expected_value(win_prob, pot_size, bet_to_call)
                if ev > 0 or win_prob > 0.3:  # Call if +EV or decent equity
                    return ActionType.CALL, ChipCount(0)
                else:
                    return ActionType.FOLD, ChipCount(0)
            else:
                return ActionType.CHECK, ChipCount(0)


# --- Factory ---


def load_personalities(config_path: str) -> Dict[str, Personality]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    personalities = {}
    for p in data:
        personalities[p["name"]] = Personality(
            name=p["name"],
            vpip=p["vpip"],
            pfr=p["pfr"],
            aggression_factor=p["aggression_factor"],
            bluff_frequency=p["bluff_frequency"],
            aggression=p["aggression_factor"],
        )
    return personalities

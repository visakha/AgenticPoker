from typing import List, Dict
from engine.domain import (
    GameState,
    Player,
    PlayerID,
    ChipCount,
    PlayerStatus,
    HandPhase,
    Card,
    Rank,
    Suit,
    Pot,
)
from engine.logic import apply_action, create_deck, shuffle_deck, draw_cards, ActionType


class BDDAgentRunner:
    """
    Acts as a 'God Agent' / Tester for BDD scenarios.
    Can inject state, force actions, and inspect results.
    """

    def __init__(self):
        self.game_state: GameState = None
        self.players: Dict[str, PlayerID] = {}

    def setup_game(self, player_names: List[str]):
        players = []
        for i, name in enumerate(player_names):
            pid = PlayerID(f"p_{i}")
            self.players[name] = pid
            players.append(
                Player(
                    id=pid,
                    name=name,
                    chips=ChipCount(1000),
                    status=PlayerStatus.ACTIVE,
                    hole_cards=[],
                    personality="neutral",
                )
            )

        self.game_state = GameState(players=players, deck_seed=42)

    def _parse_cards(self, card_str: str) -> List[Card]:
        # Parses "Ah Kh" or "Ah,Kh" or "Ah, Kh"
        card_strs = card_str.replace(",", " ").split()
        cards = []
        for cs in card_strs:
            rank_char = cs[:-1]
            suit_char = cs[-1]

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

            cards.append(Card(rank_map[rank_char], suit_map[suit_char.lower()]))
        return cards

    def inject_cards(self, player_name: str, cards_str: str):
        pid = self.players[player_name]
        cards = self._parse_cards(cards_str)

        # Recreate state with new cards for player
        new_players = []
        for p in self.game_state.players:
            if p.id == pid:
                new_players.append(
                    Player(
                        id=p.id,
                        name=p.name,
                        chips=p.chips,
                        status=p.status,
                        hole_cards=cards,
                        current_bet=p.current_bet,
                        is_human=p.is_human,
                        personality=p.personality,
                    )
                )
            else:
                new_players.append(p)

        self.game_state = GameState(
            id=self.game_state.id,
            phase=self.game_state.phase,
            community_cards=self.game_state.community_cards,
            pot=self.game_state.pot,
            current_bet=self.game_state.current_bet,
            players=new_players,
            dealer_index=self.game_state.dealer_index,
            current_player_index=self.game_state.current_player_index,
            deck_seed=self.game_state.deck_seed,
            history=self.game_state.history,
        )

    def inject_community_cards(self, cards_str: str):
        cards = self._parse_cards(cards_str)

        # Force update community cards
        self.game_state = GameState(
            id=self.game_state.id,
            phase=self.game_state.phase,
            community_cards=cards,  # Injected
            pot=self.game_state.pot,
            current_bet=self.game_state.current_bet,
            players=self.game_state.players,
            dealer_index=self.game_state.dealer_index,
            current_player_index=self.game_state.current_player_index,
            deck_seed=self.game_state.deck_seed,
            history=self.game_state.history,
        )

    def player_acts(self, player_name: str, action: str, amount: int = 0):
        pid = self.players[player_name]
        act_type = ActionType[action]

        # Apply action using the engine logic
        self.game_state = apply_action(self.game_state, pid, act_type, ChipCount(amount))

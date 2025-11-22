"""Poker-specific tokenization for LLM input."""

from typing import List, Tuple, Optional
import torch
from engine.domain import Card, Rank, Suit, GameState, Player, HandPhase, PlayerID


class PokerTokenizer:
    """
    Tokenizes poker game states into tensor representations for LLM input.
    
    Encoding scheme:
    - Cards: 52 tokens (13 ranks × 4 suits) + 1 for empty/unknown
    - Chips: Logarithmic encoding for better numerical stability
    - Position: One-hot encoding for player position
    - Phase: One-hot encoding for game phase
    - Actions: Categorical encoding
    """
    
    # Token ranges
    CARD_TOKENS = 53  # 52 cards + empty
    CHIP_BINS = 20  # Logarithmic bins for chip amounts
    POSITION_TOKENS = 9  # Max 9 players
    PHASE_TOKENS = 5  # PREFLOP, FLOP, TURN, RIVER, SHOWDOWN
    ACTION_TOKENS = 5  # FOLD, CHECK, CALL, RAISE, ALL_IN
    
    # Special tokens
    EMPTY_CARD_TOKEN = 52
    PAD_TOKEN = 0
    
    def __init__(self, max_seq_len: int = 128):
        """
        Initialize tokenizer.
        
        Args:
            max_seq_len: Maximum sequence length for action history
        """
        self.max_seq_len = max_seq_len
        
        # Create card to token mapping
        self.card_to_token = self._create_card_mapping()
        self.token_to_card = {v: k for k, v in self.card_to_token.items()}
    
    def _create_card_mapping(self) -> dict:
        """Create mapping from cards to token IDs."""
        mapping = {}
        token_id = 0
        
        # Order: Spades, Hearts, Diamonds, Clubs (standard)
        suits = [Suit.SPADES, Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS]
        ranks = [
            Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
            Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE
        ]
        
        for suit in suits:
            for rank in ranks:
                card = Card(rank, suit)
                mapping[card] = token_id
                token_id += 1
        
        return mapping
    
    def encode_card(self, card: Optional[Card]) -> int:
        """Encode a single card to token ID."""
        if card is None:
            return self.EMPTY_CARD_TOKEN
        return self.card_to_token.get(card, self.EMPTY_CARD_TOKEN)
    
    def encode_cards(self, cards: List[Card], max_cards: int = 7) -> torch.Tensor:
        """
        Encode a list of cards to tensor.
        
        Args:
            cards: List of cards to encode
            max_cards: Maximum number of cards (pad/truncate to this)
        
        Returns:
            Tensor of shape (max_cards,) with card token IDs
        """
        tokens = [self.encode_card(card) for card in cards[:max_cards]]
        # Pad with empty card tokens
        while len(tokens) < max_cards:
            tokens.append(self.EMPTY_CARD_TOKEN)
        return torch.tensor(tokens, dtype=torch.long)
    
    def encode_chips(self, chips: int) -> torch.Tensor:
        """
        Encode chip amount using logarithmic binning.
        
        Args:
            chips: Chip amount
        
        Returns:
            Tensor of shape (CHIP_BINS,) with one-hot encoding
        """
        if chips <= 0:
            bin_idx = 0
        else:
            # Logarithmic binning: log2(chips + 1)
            import math
            log_chips = math.log2(chips + 1)
            bin_idx = min(int(log_chips), self.CHIP_BINS - 1)
        
        # One-hot encoding
        encoding = torch.zeros(self.CHIP_BINS, dtype=torch.float32)
        encoding[bin_idx] = 1.0
        return encoding
    
    def encode_position(self, position: int, num_players: int) -> torch.Tensor:
        """
        Encode player position relative to dealer.
        
        Args:
            position: Player position (0 = dealer)
            num_players: Total number of players
        
        Returns:
            Tensor of shape (POSITION_TOKENS,) with one-hot encoding
        """
        encoding = torch.zeros(self.POSITION_TOKENS, dtype=torch.float32)
        if 0 <= position < self.POSITION_TOKENS:
            encoding[position] = 1.0
        return encoding
    
    def encode_phase(self, phase: HandPhase) -> torch.Tensor:
        """
        Encode game phase.
        
        Returns:
            Tensor of shape (PHASE_TOKENS,) with one-hot encoding
        """
        phase_map = {
            HandPhase.PREFLOP: 0,
            HandPhase.FLOP: 1,
            HandPhase.TURN: 2,
            HandPhase.RIVER: 3,
            HandPhase.SHOWDOWN: 4
        }
        encoding = torch.zeros(self.PHASE_TOKENS, dtype=torch.float32)
        encoding[phase_map[phase]] = 1.0
        return encoding
    
    def encode_player_state(self, player: Player, position: int, num_players: int) -> torch.Tensor:
        """
        Encode a single player's state.
        
        Returns:
            Tensor with concatenated features
        """
        # Hole cards (2 cards)
        hole_cards = self.encode_cards(player.hole_cards, max_cards=2)
        
        # Chips (one-hot)
        chips = self.encode_chips(player.chips)
        
        # Current bet (one-hot)
        current_bet = self.encode_chips(player.current_bet)
        
        # Position (one-hot)
        pos = self.encode_position(position, num_players)
        
        # Status (one-hot: active, folded, all-in, sitting out)
        status = torch.zeros(4, dtype=torch.float32)
        status_map = {"ACTIVE": 0, "FOLDED": 1, "ALL_IN": 2, "SITTING_OUT": 3}
        status[status_map.get(player.status.name, 0)] = 1.0
        
        # Concatenate all features
        return torch.cat([
            hole_cards.float(),
            chips,
            current_bet,
            pos,
            status
        ])
    
    def encode_game_state(self, state: GameState, player_id: PlayerID) -> torch.Tensor:
        """
        Encode complete game state from perspective of a specific player.
        
        Args:
            state: Current game state
            player_id: ID of the player whose perspective to encode
        
        Returns:
            Tensor with complete game state encoding
        """
        # Find player and their position
        player_idx = next(
            (i for i, p in enumerate(state.players) if p.id == player_id),
            0
        )
        player = state.players[player_idx]
        
        # Relative position (0 = self, 1 = next player, etc.)
        num_players = len(state.players)
        
        # Community cards (up to 5)
        community = self.encode_cards(state.community_cards, max_cards=5)
        
        # Pot size
        pot = self.encode_chips(state.pot.amount)
        
        # Current bet
        current_bet = self.encode_chips(state.current_bet)
        
        # Phase
        phase = self.encode_phase(state.phase)
        
        # Self state (always first)
        self_state = self.encode_player_state(player, 0, num_players)
        
        # Other players' states (relative positions)
        other_states = []
        for i in range(1, num_players):
            other_idx = (player_idx + i) % num_players
            other_player = state.players[other_idx]
            # Don't include hole cards for other players (they're hidden)
            other_player_hidden = Player(
                id=other_player.id,
                name=other_player.name,
                chips=other_player.chips,
                status=other_player.status,
                hole_cards=[],  # Hidden
                current_bet=other_player.current_bet,
                is_human=other_player.is_human,
                personality=other_player.personality,
                acted_this_round=other_player.acted_this_round
            )
            other_states.append(
                self.encode_player_state(other_player_hidden, i, num_players)
            )
        
        # Pad to max players
        while len(other_states) < self.POSITION_TOKENS - 1:
            other_states.append(torch.zeros_like(other_states[0] if other_states else self_state))
        
        # Concatenate all features
        features = torch.cat([
            community.float(),
            pot,
            current_bet,
            phase,
            self_state,
            *other_states[:self.POSITION_TOKENS - 1]
        ])
        
        return features
    
    def get_state_dim(self) -> int:
        """Get the total dimension of encoded state."""
        # This is approximate - calculate based on actual encoding
        card_dim = 5  # Community cards
        pot_dim = self.CHIP_BINS
        bet_dim = self.CHIP_BINS
        phase_dim = self.PHASE_TOKENS
        
        # Player state dimension
        player_dim = (
            2 +  # Hole cards
            self.CHIP_BINS +  # Chips
            self.CHIP_BINS +  # Current bet
            self.POSITION_TOKENS +  # Position
            4  # Status
        )
        
        # Self + max other players
        total_player_dim = player_dim * self.POSITION_TOKENS
        
        return card_dim + pot_dim + bet_dim + phase_dim + total_player_dim


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = PokerTokenizer()
    print(f"State dimension: {tokenizer.get_state_dim()}")
    
    # Test card encoding
    card = Card(Rank.ACE, Suit.SPADES)
    token = tokenizer.encode_card(card)
    print(f"Ace of Spades token: {token}")
    
    # Test chip encoding
    chips = tokenizer.encode_chips(1000)
    print(f"Chip encoding shape: {chips.shape}")

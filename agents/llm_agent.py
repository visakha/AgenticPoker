"""
LLM-based poker agent using trained transformer model.

This agent uses a trained transformer model to make poker decisions,
compatible with the existing Agent protocol.
"""

from typing import Optional, Tuple
import torch
from engine.domain import GameState, PlayerID, ActionType, ChipCount
from engine.mcp import MCPMessage, ActionResponse
from training.model import PokerTransformer, PokerTokenizer


class LLMAgent:
    """
    Agent that uses a trained LLM for decision-making.
    
    Implements the Agent protocol from the existing codebase.
    """
    
    def __init__(
        self,
        player_id: PlayerID,
        model: PokerTransformer,
        tokenizer: PokerTokenizer,
        device: str = "cpu",
        temperature: float = 1.0,
        deterministic: bool = False
    ):
        """
        Initialize LLM agent.
        
        Args:
            player_id: Unique player identifier
            model: Trained poker transformer model
            tokenizer: Poker state tokenizer
            device: Device to run model on ("cpu" or "cuda")
            temperature: Sampling temperature (higher = more exploration)
            deterministic: If True, always pick best action
        """
        self.player_id = player_id
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.deterministic = deterministic
        
        self.current_game_state: Optional[GameState] = None
        
        # Set model to eval mode by default
        self.model.eval()
    
    def set_training_mode(self, training: bool = True) -> None:
        """Set model to training or evaluation mode."""
        if training:
            self.model.train()
        else:
            self.model.eval()
    
    def receive_message(self, message: MCPMessage) -> Optional[MCPMessage]:
        """
        Process MCP message and optionally return response.
        
        Args:
            message: Incoming MCP message
        
        Returns:
            Response message if action is required, None otherwise
        """
        if message.message_type == "GAME_STATE_UPDATE":
            # Store state for decision making
            if isinstance(message.payload, dict) and "state" in message.payload:
                self.current_game_state = message.payload["state"]
            return None
        
        elif message.message_type == "ACTION_REQUEST":
            # Make decision using LLM
            action, amount = self._decide_action()
            
            return MCPMessage(
                sender=self.player_id,
                recipient="Dealer",
                message_type="ACTION_RESPONSE",
                payload=ActionResponse(
                    action_type=action,
                    amount=amount,
                    dialogue=None  # Could add LLM-generated dialogue later
                ),
            )
        
        return None
    
    def _decide_action(self) -> Tuple[ActionType, ChipCount]:
        """
        Decide action using the LLM model.
        
        Returns:
            Tuple of (action_type, amount)
        """
        if not self.current_game_state:
            # Fallback if no state
            return ActionType.CHECK, ChipCount(0)
        
        # Encode game state
        state_tensor = self.tokenizer.encode_game_state(
            self.current_game_state,
            self.player_id
        )
        state_tensor = state_tensor.unsqueeze(0).to(self.device)  # Add batch dim
        
        # Get action from model
        with torch.no_grad():
            action_idx, log_prob = self.model.sample_action(
                state_tensor,
                temperature=self.temperature,
                deterministic=self.deterministic
            )
        
        action_idx = action_idx.item()
        
        # Map action index to ActionType
        # 0: FOLD, 1: CHECK, 2: CALL, 3: RAISE, 4: ALL_IN
        action_map = [
            ActionType.FOLD,
            ActionType.CHECK,
            ActionType.CALL,
            ActionType.RAISE,
            ActionType.ALL_IN
        ]
        
        action_type = action_map[action_idx]
        
        # Determine amount based on action and game state
        amount = self._get_action_amount(action_type)
        
        return action_type, amount
    
    def _get_action_amount(self, action_type: ActionType) -> ChipCount:
        """
        Determine chip amount for the action.
        
        Args:
            action_type: Type of action
        
        Returns:
            Chip amount for the action
        """
        if not self.current_game_state:
            return ChipCount(0)
        
        # Find self in state
        me = next(
            (p for p in self.current_game_state.players if p.id == self.player_id),
            None
        )
        
        if not me:
            return ChipCount(0)
        
        if action_type == ActionType.FOLD:
            return ChipCount(0)
        
        elif action_type == ActionType.CHECK:
            return ChipCount(0)
        
        elif action_type == ActionType.CALL:
            # Call the current bet
            return ChipCount(0)  # Engine handles this
        
        elif action_type == ActionType.RAISE:
            # Raise by pot-sized amount (could be learned by model later)
            pot_size = self.current_game_state.pot.amount
            current_bet = self.current_game_state.current_bet
            raise_amount = current_bet + max(pot_size // 2, 20)
            return ChipCount(min(raise_amount, me.chips))
        
        elif action_type == ActionType.ALL_IN:
            return ChipCount(me.chips)
        
        return ChipCount(0)
    
    def get_action_with_info(
        self,
        state: GameState
    ) -> Tuple[ActionType, ChipCount, torch.Tensor, torch.Tensor]:
        """
        Get action along with log probability and value (for training).
        
        Args:
            state: Current game state
        
        Returns:
            Tuple of (action_type, amount, log_prob, value)
        """
        self.current_game_state = state
        
        # Encode state
        state_tensor = self.tokenizer.encode_game_state(state, self.player_id)
        state_tensor = state_tensor.unsqueeze(0).to(self.device)
        
        # Get action and value
        action_idx, log_prob = self.model.sample_action(
            state_tensor,
            temperature=self.temperature,
            deterministic=self.deterministic
        )
        value = self.model.get_value(state_tensor)
        
        # Map to action type
        action_map = [
            ActionType.FOLD,
            ActionType.CHECK,
            ActionType.CALL,
            ActionType.RAISE,
            ActionType.ALL_IN
        ]
        action_type = action_map[action_idx.item()]
        amount = self._get_action_amount(action_type)
        
        return action_type, amount, log_prob, value


def create_llm_agent(
    player_id: PlayerID,
    model_path: Optional[str] = None,
    device: str = "cpu",
    **kwargs
) -> LLMAgent:
    """
    Factory function to create an LLM agent.
    
    Args:
        player_id: Player identifier
        model_path: Path to trained model checkpoint (None = random init)
        device: Device to run on
        **kwargs: Additional arguments for LLMAgent
    
    Returns:
        Initialized LLM agent
    """
    from training.training_config import ModelConfig
    
    # Create tokenizer
    tokenizer = PokerTokenizer()
    
    # Create model
    config = ModelConfig()
    model = PokerTransformer(
        state_dim=tokenizer.get_state_dim(),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        num_actions=config.num_actions,
        dropout=config.dropout
    )
    
    # Load checkpoint if provided
    if model_path:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    return LLMAgent(
        player_id=player_id,
        model=model,
        tokenizer=tokenizer,
        device=device,
        **kwargs
    )


if __name__ == "__main__":
    # Test LLM agent creation
    from engine.domain import PlayerID
    
    agent = create_llm_agent(PlayerID("test_player"))
    print(f"Created LLM agent with {agent.model.count_parameters():,} parameters")

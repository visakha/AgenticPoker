"""Tests for the poker transformer model."""

import pytest
import torch
from training.model import PokerTransformer, PokerTokenizer


class TestPokerTokenizer:
    """Test poker tokenizer."""
    
    def test_tokenizer_initialization(self):
        """Test tokenizer can be initialized."""
        tokenizer = PokerTokenizer()
        assert tokenizer is not None
        assert tokenizer.get_state_dim() > 0
    
    def test_card_encoding(self):
        """Test card encoding."""
        from engine.domain import Card, Rank, Suit
        
        tokenizer = PokerTokenizer()
        card = Card(Rank.ACE, Suit.SPADES)
        token = tokenizer.encode_card(card)
        
        assert isinstance(token, int)
        assert 0 <= token < tokenizer.CARD_TOKENS
    
    def test_chip_encoding(self):
        """Test chip amount encoding."""
        tokenizer = PokerTokenizer()
        chips = tokenizer.encode_chips(1000)
        
        assert chips.shape == (tokenizer.CHIP_BINS,)
        assert torch.sum(chips) == 1.0  # One-hot encoding


class TestPokerTransformer:
    """Test poker transformer model."""
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        model = PokerTransformer(
            state_dim=200,
            d_model=128,
            n_heads=4,
            n_layers=2,
            d_ff=512,
            num_actions=5
        )
        assert model is not None
        assert model.count_parameters() > 0
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shapes."""
        model = PokerTransformer(
            state_dim=200,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_actions=5
        )
        
        batch_size = 4
        state = torch.randn(batch_size, 200)
        
        action_logits, value = model(state)
        
        assert action_logits.shape == (batch_size, 5)
        assert value.shape == (batch_size, 1)
    
    def test_action_sampling(self):
        """Test action sampling."""
        model = PokerTransformer(
            state_dim=200,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_actions=5
        )
        
        state = torch.randn(1, 200)
        action, log_prob = model.sample_action(state)
        
        assert action.shape == (1,)
        assert 0 <= action.item() < 5
        assert log_prob.shape == (1,)
    
    def test_deterministic_sampling(self):
        """Test deterministic action selection."""
        model = PokerTransformer(
            state_dim=200,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_actions=5
        )
        
        state = torch.randn(1, 200)
        
        # Sample twice with deterministic=True
        action1, _ = model.sample_action(state, deterministic=True)
        action2, _ = model.sample_action(state, deterministic=True)
        
        # Should get same action
        assert action1.item() == action2.item()
    
    def test_gradient_flow(self):
        """Test gradients flow through model."""
        model = PokerTransformer(
            state_dim=200,
            d_model=128,
            n_heads=4,
            n_layers=2,
            num_actions=5
        )
        
        state = torch.randn(2, 200)
        action_logits, value = model(state)
        
        loss = action_logits.sum() + value.sum()
        loss.backward()
        
        # Check gradients exist
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

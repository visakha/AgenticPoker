"""
Data collection and replay buffer for training.

Handles:
- Collection of game experiences (states, actions, rewards)
- Replay buffer for experience replay
- Data serialization and storage
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
from collections import deque
import pickle
from pathlib import Path

from engine.domain import GameState, PlayerID, ActionType


@dataclass
class Experience:
    """Single experience tuple for RL training."""
    
    state: torch.Tensor  # Encoded game state
    action: int  # Action index (0-4)
    reward: float  # Immediate reward
    next_state: Optional[torch.Tensor]  # Next state (None if terminal)
    done: bool  # Whether episode ended
    log_prob: float  # Log probability of action
    value: float  # Value estimate
    player_id: str  # Player who took action
    
    # Additional poker-specific info
    pot_size: int = 0
    chips_won: int = 0
    hand_phase: str = "PREFLOP"


@dataclass
class Episode:
    """Complete game episode with all experiences."""
    
    experiences: List[Experience] = field(default_factory=list)
    total_reward: float = 0.0
    num_hands: int = 0
    winner_id: Optional[str] = None
    
    def add_experience(self, exp: Experience) -> None:
        """Add experience to episode."""
        self.experiences.append(exp)
        self.total_reward += exp.reward
    
    def __len__(self) -> int:
        return len(self.experiences)


class ReplayBuffer:
    """
    Replay buffer for storing and sampling experiences.
    
    Supports:
    - Fixed-size circular buffer
    - Batch sampling
    - Priority sampling (optional)
    """
    
    def __init__(self, capacity: int = 100000, use_priority: bool = False):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
            use_priority: Whether to use prioritized experience replay
        """
        self.capacity = capacity
        self.use_priority = use_priority
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity) if use_priority else None
    
    def add(self, experience: Experience, priority: float = 1.0) -> None:
        """
        Add experience to buffer.
        
        Args:
            experience: Experience to add
            priority: Priority for sampling (if using priority)
        """
        self.buffer.append(experience)
        if self.use_priority and self.priorities is not None:
            self.priorities.append(priority)
    
    def add_episode(self, episode: Episode) -> None:
        """Add all experiences from an episode."""
        for exp in episode.experiences:
            self.add(exp)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """
        Sample a batch of experiences.
        
        Args:
            batch_size: Number of experiences to sample
        
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        if self.use_priority and self.priorities:
            # Priority sampling
            priorities = np.array(self.priorities)
            probs = priorities / priorities.sum()
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=probs
            )
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def get_all(self) -> List[Experience]:
        """Get all experiences in buffer."""
        return list(self.buffer)
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.clear()
        if self.priorities:
            self.priorities.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, path: Path) -> None:
        """Save buffer to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'buffer': list(self.buffer),
                'priorities': list(self.priorities) if self.priorities else None,
                'capacity': self.capacity,
                'use_priority': self.use_priority
            }, f)
    
    @classmethod
    def load(cls, path: Path) -> 'ReplayBuffer':
        """Load buffer from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        buffer = cls(capacity=data['capacity'], use_priority=data['use_priority'])
        buffer.buffer = deque(data['buffer'], maxlen=data['capacity'])
        if data['priorities']:
            buffer.priorities = deque(data['priorities'], maxlen=data['capacity'])
        
        return buffer


class DataCollector:
    """
    Collects training data from game simulations.
    
    Handles:
    - State encoding
    - Reward calculation
    - Episode construction
    """
    
    def __init__(
        self,
        tokenizer,
        replay_buffer: Optional[ReplayBuffer] = None,
        gamma: float = 0.99
    ):
        """
        Initialize data collector.
        
        Args:
            tokenizer: Poker state tokenizer
            replay_buffer: Optional replay buffer to store experiences
            gamma: Discount factor for rewards
        """
        self.tokenizer = tokenizer
        self.replay_buffer = replay_buffer
        self.gamma = gamma
        
        # Current episode being collected
        self.current_episodes: Dict[str, Episode] = {}
    
    def start_episode(self, player_ids: List[PlayerID]) -> None:
        """Start collecting a new episode for players."""
        for player_id in player_ids:
            self.current_episodes[player_id] = Episode()
    
    def add_experience(
        self,
        player_id: PlayerID,
        state: GameState,
        action: ActionType,
        reward: float,
        next_state: Optional[GameState],
        done: bool,
        log_prob: float = 0.0,
        value: float = 0.0
    ) -> None:
        """
        Add an experience to the current episode.
        
        Args:
            player_id: Player who took action
            state: Game state before action
            action: Action taken
            reward: Reward received
            next_state: State after action (None if terminal)
            done: Whether episode ended
            log_prob: Log probability of action
            value: Value estimate
        """
        # Encode states
        state_tensor = self.tokenizer.encode_game_state(state, player_id)
        next_state_tensor = (
            self.tokenizer.encode_game_state(next_state, player_id)
            if next_state else None
        )
        
        # Map action to index
        action_map = {
            ActionType.FOLD: 0,
            ActionType.CHECK: 1,
            ActionType.CALL: 2,
            ActionType.RAISE: 3,
            ActionType.ALL_IN: 4
        }
        action_idx = action_map.get(action, 1)  # Default to CHECK
        
        # Create experience
        exp = Experience(
            state=state_tensor,
            action=action_idx,
            reward=reward,
            next_state=next_state_tensor,
            done=done,
            log_prob=log_prob,
            value=value,
            player_id=str(player_id),
            pot_size=state.pot.amount,
            hand_phase=state.phase.name
        )
        
        # Add to current episode
        if str(player_id) in self.current_episodes:
            self.current_episodes[str(player_id)].add_experience(exp)
    
    def end_episode(
        self,
        player_id: PlayerID,
        final_chips: int,
        starting_chips: int
    ) -> Episode:
        """
        End episode and calculate returns.
        
        Args:
            player_id: Player whose episode is ending
            final_chips: Final chip count
            starting_chips: Starting chip count
        
        Returns:
            Completed episode
        """
        episode = self.current_episodes.get(str(player_id))
        if not episode:
            return Episode()
        
        # Calculate total reward (profit/loss)
        total_reward = final_chips - starting_chips
        episode.total_reward = total_reward
        
        # Compute discounted returns (backward pass)
        returns = []
        G = 0.0
        for exp in reversed(episode.experiences):
            G = exp.reward + self.gamma * G
            returns.insert(0, G)
        
        # Update experiences with returns
        for exp, G in zip(episode.experiences, returns):
            exp.reward = G  # Replace immediate reward with return
        
        # Add to replay buffer if available
        if self.replay_buffer:
            self.replay_buffer.add_episode(episode)
        
        # Clear current episode
        del self.current_episodes[str(player_id)]
        
        return episode
    
    def get_batch(self, batch_size: int) -> Optional[Tuple]:
        """
        Get a batch of experiences for training.
        
        Args:
            batch_size: Size of batch
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, log_probs, values)
        """
        if not self.replay_buffer or len(self.replay_buffer) < batch_size:
            return None
        
        experiences = self.replay_buffer.sample(batch_size)
        
        # Stack into tensors
        states = torch.stack([exp.state for exp in experiences])
        actions = torch.tensor([exp.action for exp in experiences], dtype=torch.long)
        rewards = torch.tensor([exp.reward for exp in experiences], dtype=torch.float32)
        
        # Handle next states (some may be None)
        next_states = []
        dones = []
        for exp in experiences:
            if exp.next_state is not None:
                next_states.append(exp.next_state)
                dones.append(0.0)
            else:
                next_states.append(torch.zeros_like(exp.state))
                dones.append(1.0)
        
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        log_probs = torch.tensor([exp.log_prob for exp in experiences], dtype=torch.float32)
        values = torch.tensor([exp.value for exp in experiences], dtype=torch.float32)
        
        return states, actions, rewards, next_states, dones, log_probs, values


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    print(f"Created replay buffer with capacity {buffer.capacity}")
    
    # Test data collector
    from training.model import PokerTokenizer
    tokenizer = PokerTokenizer()
    collector = DataCollector(tokenizer, buffer)
    print("Created data collector")

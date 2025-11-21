# LLM Training Infrastructure

## Overview

This directory contains the complete infrastructure for training small LLM agents to play poker through self-play reinforcement learning.

## Architecture

### Core Components

1. **Model** (`model/`)
   - `model.py`: Transformer-based poker LLM (~15M parameters)
   - `tokenizer.py`: Poker-specific state encoding

2. **Training** 
   - `trainer.py`: PPO (Proximal Policy Optimization) implementation
   - `train.py`: Main training orchestration script
   - `training_config.py`: Configuration dataclasses

3. **Data Collection**
   - `data_collector.py`: Experience collection and replay buffer
   - `game_simulator.py`: High-performance parallel game execution

4. **Evaluation**
   - `evaluator.py`: Model evaluation against baselines
   - `metrics.py`: Training metrics and TensorBoard logging

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Training

```bash
# Train with default settings (1M games)
make train

# Or with custom parameters
python training/train.py --num-games 100000 --parallel-games 4 --device cuda
```

### 3. Monitor Training

```bash
# Open TensorBoard
tensorboard --logdir=training/logs
```

### 4. Evaluate Model

```bash
# Evaluate final model
make evaluate

# Or evaluate specific checkpoint
python training/evaluator.py training/checkpoints/checkpoint_step_5000.pt --num-games 100
```

## Training Configuration

Edit `config/training_config.yaml` to customize:

- **Model Architecture**: `d_model`, `n_layers`, `n_heads`
- **Training Hyperparameters**: `learning_rate`, `batch_size`, `gamma`
- **Simulation Settings**: `num_players`, `parallel_games`
- **Checkpointing**: `checkpoint_frequency`, `keep_checkpoints`

## Command Line Options

```bash
python training/train.py --help
```

Key options:
- `--num-games`: Total games to play (default: 1,000,000)
- `--parallel-games`: Number of parallel games (default: 8)
- `--batch-size`: Training batch size (default: 256)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--device`: Device to use (cuda/cpu)
- `--resume`: Resume from latest checkpoint
- `--checkpoint`: Path to specific checkpoint

## Directory Structure

```
training/
├── model/
│   ├── __init__.py
│   ├── model.py          # Transformer architecture
│   └── tokenizer.py      # State encoding
├── __init__.py
├── train.py              # Main training script
├── trainer.py            # PPO trainer
├── training_config.py    # Configuration
├── data_collector.py     # Data collection
├── game_simulator.py     # Game simulation
├── evaluator.py          # Model evaluation
├── metrics.py            # Metrics tracking
├── checkpoints/          # Saved models (created during training)
├── logs/                 # TensorBoard logs (created during training)
└── data/                 # Training data (created during training)
```

## Training Process

1. **Data Collection**: Simulator runs parallel poker games with current model
2. **Experience Storage**: States, actions, rewards stored in replay buffer
3. **Training Update**: PPO updates model using collected experiences
4. **Evaluation**: Periodic evaluation against baseline agents
5. **Checkpointing**: Regular model saves for recovery

## Model Architecture

- **Input**: Encoded game state (cards, chips, pot, positions)
- **Encoder**: 6-layer transformer with 8 attention heads
- **Output Heads**:
  - Policy head: Action probabilities (fold/check/call/raise/all-in)
  - Value head: State value estimation

## Performance

Expected performance on modern hardware:
- **CPU**: ~50-100 games/second
- **GPU**: ~200-500 games/second
- **1M games**: 3-10 hours depending on hardware

## Checkpoints

Checkpoints are saved to `training/checkpoints/`:
- `checkpoint_step_N.pt`: Regular checkpoints during training
- `final_model.pt`: Final trained model

Load a checkpoint:
```python
from agents.llm_agent import create_llm_agent

agent = create_llm_agent(
    player_id="llm_player",
    model_path="training/checkpoints/final_model.pt",
    device="cuda"
)
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in config
- Reduce `parallel_games`
- Use smaller model (`--d-model 128 --n-layers 4`)

### Slow Training
- Enable multiprocessing: `use_multiprocessing: true` in config
- Increase `parallel_games`
- Use GPU: `--device cuda`

### Poor Performance
- Increase training games
- Adjust learning rate
- Tune PPO hyperparameters (`clip_epsilon`, `entropy_coef`)

## Testing

Run tests:
```bash
pytest tests/test_model.py tests/test_simulator.py tests/test_training.py -v
```

## References

- PPO Algorithm: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
- Transformer Architecture: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

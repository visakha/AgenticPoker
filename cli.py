import argparse
import json
import random
import time
from typing import List
from engine.domain import GameState, Player, PlayerID, ChipCount, PlayerStatus, HandPhase
from engine.logic import create_deck, shuffle_deck, draw_cards, next_player_index, apply_action
from engine.mcp import MCPMessage, ActionResponse, ActionRequest, GameStateUpdate
from agents.functional_agent import FunctionalAgent, load_personalities, Personality


def run_simulation(num_games: int, log_file: str, config_path: str):
    print(f"Starting simulation of {num_games} games...")
    print(f"Loading personalities from {config_path}...")

    personalities = load_personalities(config_path)
    personality_names = list(personalities.keys())

    # Create Agents
    agents: List[FunctionalAgent] = []
    players: List[Player] = []

    for i in range(6):
        p_name = personality_names[i % len(personality_names)]
        p_id = PlayerID(f"bot_{i}")
        agent = FunctionalAgent(p_id, personalities[p_name])
        agents.append(agent)
        players.append(
            Player(id=p_id, name=f"{p_name}_{i}", chips=ChipCount(1000), personality=p_name)
        )

    # Open log file
    with open(log_file, "w") as f:
        for game_idx in range(num_games):
            # Init Game
            deck = shuffle_deck(create_deck(), seed=random.randint(0, 1000000))

            # Deal Hole Cards
            updated_players = []
            for p in players:
                cards, deck = draw_cards(deck, 2)
                updated_players.append(
                    Player(
                        id=p.id,
                        name=p.name,
                        chips=p.chips,
                        status=PlayerStatus.ACTIVE,
                        hole_cards=cards,
                        personality=p.personality,
                    )
                )

            state = GameState(players=updated_players, deck_seed=random.randint(0, 1000))

            # Simple Game Loop (Pre-flop only for prototype)
            # In a real engine, we'd loop through phases

            game_log = {
                "game_id": str(state.id),
                "players": [p.name for p in state.players],
                "actions": [],
            }

            # Simulate one round of betting
            active_players = [p for p in state.players if p.status == PlayerStatus.ACTIVE]
            for _ in range(len(active_players)):
                current_idx = state.current_player_index
                current_player = state.players[current_idx]

                # 1. Dealer -> Agent: Update
                update_msg = MCPMessage(
                    "Dealer", current_player.id, "GAME_STATE_UPDATE", {"state": state}
                )
                # Find agent
                agent = next(a for a in agents if a.player_id == current_player.id)
                agent.receive_message(update_msg)

                # 2. Dealer -> Agent: Request Action
                req_msg = MCPMessage("Dealer", current_player.id, "ACTION_REQUEST", {})
                resp_msg = agent.receive_message(req_msg)

                if resp_msg and resp_msg.message_type == "ACTION_RESPONSE":
                    payload: ActionResponse = resp_msg.payload

                    # 3. Apply Action
                    state = apply_action(
                        state, current_player.id, payload.action_type, payload.amount
                    )

                    # Log
                    action_entry = {
                        "player": current_player.name,
                        "action": payload.action_type.name,
                        "amount": payload.amount,
                        "phase": state.phase.name,
                        "dialogue": payload.dialogue,
                    }
                    game_log["actions"].append(action_entry)
                    dialogue_str = f' "{payload.dialogue}"' if payload.dialogue else ""
                    print(
                        f"Game {game_idx}: {current_player.name} did {payload.action_type.name}{dialogue_str}"
                    )

            # End of Game (Mock)
            f.write(json.dumps(game_log) + "\n")

    print(f"Simulation complete. Logs saved to {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic Poker CLI Runner")
    parser.add_argument("--games", type=int, default=10, help="Number of games to simulate")
    parser.add_argument("--log-file", type=str, default="game_logs.jsonl", help="Output log file")
    parser.add_argument(
        "--config",
        type=str,
        default="config/tune_ai_players.yml",
        help="Path to personality config",
    )

    args = parser.parse_args()
    run_simulation(args.games, args.log_file, args.config)

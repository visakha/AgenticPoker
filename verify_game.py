import requests
import time
import json

BASE_URL = "http://127.0.0.1:8000/api/game"

def print_state(state, label):
    players = state.get("players", [])
    current_player = next((p for p in players if p.get("is_current")), None)
    pot = state.get("pot", 0)
    phase = state.get("phase", "UNKNOWN")
    
    print(f"--- {label} ---")
    print(f"Phase: {phase}")
    print(f"Pot: {pot}")
    if current_player:
        print(f"Current Player: {current_player['name']}")
        print(f"Current Bet: {current_player['current_bet']}")
    else:
        print("Current Player: None")
    print("-" * 20)

def main():
    # New Game
    print("Starting new game...")
    resp = requests.post(f"{BASE_URL}/new")
    if resp.status_code != 200:
        print(f"Failed to start game: {resp.text}")
        return
    state = resp.json()
    print_state(state, "Initial State")
    
    # Run 50 steps
    for i in range(50):
        print(f"Running step {i+1}...")
        resp = requests.post(f"{BASE_URL}/step")
        if resp.status_code != 200:
            print(f"Step failed: {resp.text}")
            break
        state = resp.json()
        print_state(state, f"Step {i+1}")
        time.sleep(0.5)

if __name__ == "__main__":
    main()

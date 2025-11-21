import streamlit as st
import time
import random
from typing import List
from engine.domain import GameState, Player, PlayerID, ChipCount, PlayerStatus, HandPhase, ActionType
from engine.logic import create_deck, shuffle_deck, draw_cards, apply_action
from agents.functional_agent import FunctionalAgent, load_personalities, Personality
from engine.mcp import MCPMessage, ActionResponse

# --- Setup & Config ---
st.set_page_config(layout="wide", page_title="Agentic Poker", page_icon="🃏")

# Load Personalities
if "personalities" not in st.session_state:
    try:
        st.session_state.personalities = load_personalities("config/tune_ai_players.yml")
    except:
        # Fallback if config missing
        st.session_state.personalities = {}

# --- Game State Management ---

def init_game():
    # Create Players
    players = []
    agent_objs = []
    
    p_names = list(st.session_state.personalities.keys())
    if not p_names: p_names = ["Bot1", "Bot2", "Bot3", "Bot4", "Bot5", "Bot6"]
    
    for i in range(6):
        name = p_names[i % len(p_names)]
        pid = PlayerID(f"p_{i}")
        
        # Human Player (Player 0)
        is_human = (i == 0) and (st.session_state.game_mode == "Play")
        
        p = Player(
            id=pid, name=f"{name} (Human)" if is_human else name, 
            chips=ChipCount(1000), status=PlayerStatus.ACTIVE,
            is_human=is_human, personality=name
        )
        players.append(p)
        
        # Create Agent for bots
        if not is_human:
            pers = st.session_state.personalities.get(name, None)
            # Create dummy personality if missing
            if not pers:
                pers = Personality(
                    name=name, vpip=0.5, pfr=0.5, 
                    aggression_factor=0.5, bluff_frequency=0.5, 
                    aggression=0.5
                )
            agent = FunctionalAgent(pid, pers)
            agent_objs.append(agent)
        else:
            agent_objs.append(None) # Placeholder for human

    # Init State
    deck = shuffle_deck(create_deck(), seed=random.randint(0, 1000000))
    
    # Deal Hole Cards
    updated_players = []
    for p in players:
        cards, deck = draw_cards(deck, 2)
        updated_players.append(Player(
            id=p.id, name=p.name, chips=p.chips, status=p.status,
            hole_cards=cards, current_bet=p.current_bet,
            is_human=p.is_human, personality=p.personality
        ))

    st.session_state.game_state = GameState(players=updated_players, deck_seed=random.randint(0, 1000))
    st.session_state.agents = agent_objs
    st.session_state.logs = []
    st.session_state.running = False

if "game_state" not in st.session_state:
    st.session_state.game_mode = "Watch" # Default
    init_game()

# --- Game Loop Step ---

def run_step():
    state = st.session_state.game_state
    agents = st.session_state.agents
    
    # Check if game over (Showdown)
    if state.phase == HandPhase.SHOWDOWN:
        st.toast("Hand Over! Starting new hand in 3s...")
        time.sleep(3)
        init_game() # Reset
        st.rerun()
        return

    current_idx = state.current_player_index
    current_player = state.players[current_idx]
    
    # Check if human turn
    if current_player.is_human:
        st.session_state.waiting_for_human = True
        return

    # Agent Turn
    agent = agents[current_idx]
    if agent:
        # 1. Update Agent
        agent.current_game_state = state
        
        # 2. Request Action
        resp_msg = agent.receive_message(MCPMessage("Dealer", current_player.id, "ACTION_REQUEST", {}))
        
        if resp_msg and resp_msg.message_type == "ACTION_RESPONSE":
            payload: ActionResponse = resp_msg.payload
            
            # Apply
            new_state = apply_action(state, current_player.id, payload.action_type, payload.amount)
            st.session_state.game_state = new_state
            
            # Log
            log_entry = {
                "sender": current_player.name,
                "message": f"{payload.action_type.name} {payload.amount if payload.amount > 0 else ''}",
                "type": "game"
            }
            if payload.dialogue:
                 st.session_state.logs.append({
                    "sender": current_player.name,
                    "message": payload.dialogue,
                    "type": "chat"
                })
            st.session_state.logs.append(log_entry)

# --- UI ---

import streamlit.components.v1 as components
import os

# Declare component
_component_func = components.declare_component(
    "poker_component",
    path=os.path.join(os.path.dirname(__file__), "poker_component/dist")
)

def poker_component(game_state, key=None):
    return _component_func(game_state=game_state, key=key)

# Sidebar
with st.sidebar:
    st.title("Agentic Poker 🤖")
    
    mode = st.radio("Mode", ["Watch", "Play"], index=0 if st.session_state.game_mode == "Watch" else 1)
    if mode != st.session_state.game_mode:
        st.session_state.game_mode = mode
        init_game()
        st.rerun()
        
    if st.button("New Hand"):
        init_game()
        st.rerun()
        
    if st.button("Step"):
        run_step()
        st.rerun()
        
    auto_play = st.checkbox("Auto Play", value=False)

# Main Area
if auto_play and not st.session_state.get("waiting_for_human", False):
    run_step()
    time.sleep(0.5) # Delay for visual
    st.rerun()

# Prepare State for React
# We need to serialize the state for the React component
gs = st.session_state.game_state
react_state = {
    "players": [
        {
            "name": p.name,
            "chips": p.chips,
            "cards": [{"rank": c.rank.value, "suit": c.suit.value} for c in p.hole_cards] if (st.session_state.game_mode == "Watch" or p.is_human or p.status == PlayerStatus.SHOWDOWN) else [], # Hide cards logic
            "status": p.status.name,
            "personality": p.personality,
            "vpip": 0.0, # Placeholder
            "pfr": 0.0,
            "aggression": 0.0
        }
        for p in gs.players
    ],
    "communityCards": [{"rank": c.rank.value, "suit": c.suit.value} for c in gs.community_cards],
    "pot": gs.pot.amount,
    "logs": st.session_state.logs[-10:] # Last 10 logs
}

# Render React Component
action = poker_component(game_state=react_state)

# Handle Human Action from React (if implemented in React to send events)
if action:
    # TODO: Handle human action from UI
    pass

# Human Controls (Streamlit fallback)
if st.session_state.get("waiting_for_human", False):
    st.info("Your Turn!")
    col1, col2, col3 = st.columns(3)
    if col1.button("Fold"):
        st.session_state.game_state = apply_action(gs, gs.players[gs.current_player_index].id, ActionType.FOLD)
        st.session_state.waiting_for_human = False
        st.rerun()
    if col2.button("Check/Call"):
        st.session_state.game_state = apply_action(gs, gs.players[gs.current_player_index].id, ActionType.CALL) # Logic handles Check vs Call
        st.session_state.waiting_for_human = False
        st.rerun()
    if col3.button("Raise 20"):
        st.session_state.game_state = apply_action(gs, gs.players[gs.current_player_index].id, ActionType.RAISE, ChipCount(gs.current_bet + 20))
        st.session_state.waiting_for_human = False
        st.rerun()

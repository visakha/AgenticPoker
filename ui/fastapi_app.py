"""
FastAPI Poker Application

High-performance poker game with real-time updates via WebSocket.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Dict, List
import asyncio
import random
import json
from pathlib import Path

from engine.domain import GameState, Player, PlayerID, ChipCount, PlayerStatus, HandPhase, ActionType, Card
from engine.logic import create_deck, shuffle_deck, draw_cards, apply_action
from agents.functional_agent import load_personalities, Personality
from agents.llm_agent import create_llm_agent
from engine.mcp import MCPMessage, ActionResponse

# Initialize FastAPI app
app = FastAPI(title="Agentic Poker", version="2.0")

# Setup templates and static files
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# Global game state (in production, use Redis or similar)
game_sessions: Dict[str, dict] = {}

# Load personalities
try:
    personalities = load_personalities("config/tune_ai_players.yml")
except Exception as e:
    print(f"Could not load personalities: {e}")
    personalities = {
        "Tight": Personality(name="Tight", vpip=0.2, pfr=0.15, aggression_factor=0.3, bluff_frequency=0.05, aggression=0.3),
        "Loose": Personality(name="Loose", vpip=0.5, pfr=0.3, aggression_factor=0.6, bluff_frequency=0.2, aggression=0.6),
        "Aggressive": Personality(name="Aggressive", vpip=0.35, pfr=0.25, aggression_factor=0.8, bluff_frequency=0.15, aggression=0.8),
    }

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Helper functions
def card_to_str(card: Card) -> str:
    """Convert card to string representation."""
    suit_symbols = {"HEARTS": "♥️", "DIAMONDS": "♦️", "CLUBS": "♣️", "SPADES": "♠️"}
    return f"{card.rank.value}{suit_symbols.get(card.suit.name, card.suit.name[0])}"

def init_game(session_id: str = "default"):
    """Initialize a new poker game."""
    players = []
    agent_objs = []
    
    p_names = list(personalities.keys())
    if not p_names:
        p_names = ["Bot1", "Bot2", "Bot3", "Bot4", "Bot5", "Bot6"]
    
    for i in range(6):
        if i < len(p_names):
            name = p_names[i]
        else:
            name = f"Bot{i+1}"
        
        pid = PlayerID(f"p_{i}")
        
        p = Player(
            id=pid,
            name=name,
            chips=ChipCount(1000),
            status=PlayerStatus.ACTIVE,
            personality=name
        )
        players.append(p)
        
        # Create Agent
        pers = personalities.get(name)
        if not pers:
            pers = Personality(name=name, vpip=0.5, pfr=0.5, aggression_factor=0.5, bluff_frequency=0.5, aggression=0.5)
        
        # Create LLM Agent
        # We use the same model for all bots for now, but they have different personalities (which LLM might ignore for now unless conditioned)
        # Note: The current LLM agent doesn't explicitly take personality params in __init__ but we can pass them if we update it.
        # For now, we just use the trained model.
        try:
            agent = create_llm_agent(
                player_id=pid,
                model_path="training/checkpoints/final_model.pt",
                device="cpu"
            )
        except Exception as e:
            print(f"Failed to create LLM agent for {name}, falling back to FunctionalAgent: {e}")
            from agents.functional_agent import FunctionalAgent
            agent = FunctionalAgent(pid, pers)
            
        agent_objs.append(agent)
    
    # Init State
    deck = shuffle_deck(create_deck(), seed=random.randint(0, 1000000))
    
    # Deal Hole Cards
    updated_players = []
    for p in players:
        cards, deck = draw_cards(deck, 2)
        updated_players.append(Player(
            id=p.id, name=p.name, chips=p.chips, status=p.status,
            hole_cards=cards, current_bet=p.current_bet, personality=p.personality
        ))
    
    game_state = GameState(players=updated_players, deck_seed=random.randint(0, 1000))
    
    game_sessions[session_id] = {
        "game_state": game_state,
        "agents": agent_objs,
        "logs": [],
        "hand_count": 0,
        "auto_play": False,
        "dealer_button_position": 0  # Dealer button starts at position 0
    }
    
    return game_sessions[session_id]

def run_step(session_id: str = "default"):
    """Execute one step of the game."""
    session = game_sessions.get(session_id)
    if not session:
        session = init_game(session_id)
    
    state = session["game_state"]
    agents = session["agents"]
    
    # Check if game over (Showdown)
    if state.phase == HandPhase.SHOWDOWN:
        session["logs"].append({"sender": "Dealer", "message": "Hand complete! Starting new hand...", "type": "system"})
        session["hand_count"] += 1
        # Rotate dealer button to next player
        session["dealer_button_position"] = (session.get("dealer_button_position", 0) + 1) % 6
        init_game(session_id)
        return session
    
    current_idx = state.current_player_index
    current_player = state.players[current_idx]
    
    # Agent Turn
    agent = agents[current_idx]
    if agent:
        # Update Agent
        agent.current_game_state = state
        
        # Request Action
        resp_msg = agent.receive_message(MCPMessage("Dealer", current_player.id, "ACTION_REQUEST", {}))
        
        if resp_msg and resp_msg.message_type == "ACTION_RESPONSE":
            payload: ActionResponse = resp_msg.payload
            
            # Apply
            new_state = apply_action(state, current_player.id, payload.action_type, payload.amount)
            session["game_state"] = new_state
            
            # Log
            action_str = f"{payload.action_type.name}"
            if payload.amount > 0:
                action_str += f" ${payload.amount}"
            
            phase_name = state.phase.name if state.phase else "PREFLOP"
            
            session["logs"].append({
                "sender": current_player.name,
                "message": action_str,
                "type": "action",
                "phase": phase_name
            })
            
            if payload.dialogue:
                session["logs"].append({
                    "sender": current_player.name,
                    "message": payload.dialogue,
                    "type": "chat",
                    "phase": phase_name
                })
    
    return session

def get_player_actions(player_name: str, logs: list) -> list:
    """Get the last 3 actions for a specific player."""
    player_logs = [log for log in logs if log.get("sender") == player_name and log.get("type") == "action"]
    return player_logs[-3:]

def serialize_game_state(session: dict) -> dict:
    """Convert game state to JSON-serializable format."""
    state = session["game_state"]
    
    dealer_position = session.get("dealer_button_position", 0)
    
    players_data = []
    for i, player in enumerate(state.players):
        is_current = (i == state.current_player_index)
        is_dealer = (i == dealer_position)
        recent_actions = get_player_actions(player.name, session["logs"])
        
        players_data.append({
            "name": player.name,
            "chips": player.chips,
            "current_bet": player.current_bet,
            "status": player.status.name,
            "is_current": is_current,
            "is_dealer": is_dealer,
            "cards": [card_to_str(c) for c in player.hole_cards] if player.hole_cards else [],
            "recent_actions": recent_actions
        })
    
    return {
        "players": players_data,
        "community_cards": [card_to_str(c) for c in state.community_cards],
        "pot": state.pot.amount,
        "current_bet": state.current_bet,
        "phase": state.phase.name if state.phase else "PREFLOP",
        "hand_count": session["hand_count"],
        "auto_play": session.get("auto_play", False)
    }

# Routes
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main poker table page."""
    session_id = "default"
    if session_id not in game_sessions:
        init_game(session_id)
    
    game_data = serialize_game_state(game_sessions[session_id])
    return templates.TemplateResponse("poker_table.html", {
        "request": request,
        "game": game_data
    })

@app.post("/api/game/new")
async def new_hand():
    """Start a new hand."""
    session = init_game("default")
    await manager.broadcast({"type": "game_update", "data": serialize_game_state(session)})
    return serialize_game_state(session)

@app.post("/api/game/step")
async def step():
    """Execute one game step."""
    session = run_step("default")
    game_data = serialize_game_state(session)
    await manager.broadcast({"type": "game_update", "data": game_data})
    return game_data

@app.post("/api/game/auto-play")
async def toggle_auto_play(request: Request):
    """Toggle auto-play mode."""
    data = await request.json()
    enabled = data.get("enabled", False)
    
    session = game_sessions.get("default")
    if not session:
        session = init_game("default")
    
    session["auto_play"] = enabled
    
    # Start auto-play loop if enabled
    if enabled:
        asyncio.create_task(auto_play_loop())
    
    return {"auto_play": enabled}

async def auto_play_loop():
    """Auto-play game loop."""
    session = game_sessions.get("default")
    if not session:
        return
    
    while session.get("auto_play", False):
        run_step("default")
        game_data = serialize_game_state(session)
        await manager.broadcast({"type": "game_update", "data": game_data})
        await asyncio.sleep(0.05)  # 20 steps per second

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_text(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

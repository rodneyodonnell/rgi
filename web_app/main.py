# web_app/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rgi.core.game_registry import GAME_REGISTRY
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer
from typing import Dict, Any

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# In-memory storage for game sessions
games: Dict[int, Dict[str, Any]] = {}
game_counter = 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/games/new")
async def create_game(request: Request):
    global game_counter
    data = await request.json()
    game_type = data.get("game_type")
    ai_player = data.get("ai_player", False)
    registry_entry = GAME_REGISTRY.get(game_type)
    if not registry_entry.game_fn or not registry_entry.serializer_fn:
        raise HTTPException(status_code=400, detail="Invalid game type")

    game = registry_entry.game_fn()
    game_serializer = registry_entry.serializer_fn()
    state = game.initial_state()
    game_counter += 1

    # Initialize players
    players = {
        1: MinimaxPlayer(game, player_id=1) if ai_player else RandomPlayer(),
        2: RandomPlayer(),
    }

    # Store game session
    games[game_counter] = {
        "game": game,
        "serializer": game_serializer,
        "state": state,
        "players": players,
        "ai_player": ai_player,
    }
    return {"game_id": game_counter, "game_type": game_type}


@app.get("/games/{game_id}/state")
async def get_game_state(game_id: int):
    game_session = games.get(game_id)
    if not game_session:
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    serializer = game_session["serializer"]
    state = game_session["state"]
    game_state = serializer.serialize_state(game, state)

    if game_state["is_terminal"]:
        all_players = game.all_player_ids(state)
        rewards = {player_id: game.reward(state, player_id) for player_id in all_players}
        winner = [player_id for player_id, reward in rewards.items() if reward == 1.0]
        game_state["winner"] = winner[0] if winner else None
    else:
        game_state["winner"] = None

    return game_state


@app.post("/games/{game_id}/move")
async def make_move(game_id: int, action_data: Dict[str, Any]):
    game_session = games.get(game_id)
    if not game_session:
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    serializer = game_session["serializer"]
    state = game_session["state"]
    players = game_session["players"]
    ai_player = game_session["ai_player"]

    try:
        action = serializer.parse_action(game, action_data)
        if action not in game.legal_actions(state):
            raise ValueError("Invalid move")
        state = game.next_state(state, action)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    # Update game state
    game_session["state"] = state

    # AI Move
    if ai_player and not game.is_terminal(state):
        current_player_id = game.current_player_id(state)
        ai = players.get(current_player_id)
        if ai:
            ai_action = ai.select_action(state, game.legal_actions(state))
            try:
                state = game.next_state(state, ai_action)
                game_session["state"] = state
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))

    return {"success": True}


@app.get("/{game_type}/{game_id}", response_class=HTMLResponse)
async def serve_game_page(request: Request, game_type: str, game_id: int):
    if game_type not in GAME_REGISTRY:
        raise HTTPException(status_code=404, detail="Game not found")
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")

    return templates.TemplateResponse("game.html", {"request": request, "game_type": game_type, "game_id": game_id})

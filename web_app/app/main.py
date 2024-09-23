# web_app/main.py

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rgi.core.game_registry import GAME_REGISTRY
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer
from rgi.players.human_player import HumanPlayer
from typing import Dict, Any


from datetime import datetime

print("Server restarted at", datetime.now())

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# In-memory storage for game sessions
games: Dict[int, Dict[str, Any]] = {}
game_counter = 0


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    logger.debug("Serving root page")
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/games/new")
async def create_game(request: Request):
    global game_counter
    data = await request.json()
    game_type = data.get("game_type")
    player1_type = data.get("options").get("player1_type")
    player2_type = data.get("options").get("player2_type")
    logger.info(f"Creating new game. Type: {game_type}, Player 1: {player1_type}, Player 2: {player2_type}")

    registry_entry = GAME_REGISTRY.get(game_type)
    if not registry_entry:
        logger.error(f"Invalid game type: {game_type}")
        raise HTTPException(status_code=400, detail="Invalid game type")

    game = registry_entry.game_fn()
    game_serializer = registry_entry.serializer_fn()
    state = game.initial_state()
    game_counter += 1

    # Initialize players
    players = {
        1: create_player(player1_type, game, player_id=1),
        2: create_player(player2_type, game, player_id=2),
    }
    logger.debug(f"Players initialized. Player 1: {type(players[1]).__name__}, Player 2: {type(players[2]).__name__}")

    # Store game session
    games[game_counter] = {
        "game": game,
        "serializer": game_serializer,
        "state": state,
        "players": players,
        "options": {
            "player1_type": player1_type,
            "player2_type": player2_type,
        },
    }
    logger.info(f"New game created. ID: {game_counter}, Player 1: {player1_type}, Player 2: {player2_type}")
    return {"game_id": game_counter, "game_type": game_type}


def create_player(player_type: str, game, player_id: int):
    if player_type == "human":
        return HumanPlayer(game)
    elif player_type == "random":
        return RandomPlayer()
    elif player_type == "minimax":
        return MinimaxPlayer(game, player_id)
    else:
        logger.error(f"Unknown player type: {player_type}")
        raise ValueError(f"Unknown player type: {player_type}")


@app.get("/games/{game_id}/state")
async def get_game_state(game_id: int):
    logger.debug(f"Fetching game state for game ID: {game_id}")
    game_session = games.get(game_id)
    if not game_session:
        logger.error(f"Game not found. ID: {game_id}")
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    serializer = game_session["serializer"]
    state = game_session["state"]
    game_state = serializer.serialize_state(game, state)

    # Add AI player information to the game state
    game_state["options"] = game_session["options"]

    if game.is_terminal(state):
        all_players = game.all_player_ids(state)
        rewards = {player_id: game.reward(state, player_id) for player_id in all_players}
        winner = [player_id for player_id, reward in rewards.items() if reward == 1.0]
        game_state["winner"] = winner[0] if winner else None
        logger.info(f"Game {game_id} is terminal. Winner: {game_state['winner']}")
    else:
        game_state["winner"] = None

    logger.debug(f"Game state for game {game_id}: {game_state}")
    return game_state


@app.post("/games/{game_id}/move")
async def make_move(game_id: int, action_data: Dict[str, Any]):
    game_session = games.get(game_id)
    if not game_session:
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    state = game_session["state"]

    try:
        action = game_session["serializer"].parse_action(game, action_data)
        if action not in game.legal_actions(state):
            return {"success": False, "error": "Invalid move"}
        new_state = game.next_state(state, action)
        game_session["state"] = new_state
        return {"success": True}
    except ValueError as ve:
        return {"success": False, "error": str(ve)}


@app.post("/games/{game_id}/ai_move")
async def make_ai_move(game_id: int):
    logger.debug(f"Attempting AI move for game ID: {game_id}")
    game_session = games.get(game_id)
    if not game_session:
        logger.error(f"Game not found. ID: {game_id}")
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    state = game_session["state"]
    players = game_session["players"]

    current_player_id = game.current_player_id(state)
    current_player = players[current_player_id]

    if game.is_terminal(state):
        logger.info(f"Game {game_id} is already in a terminal state.")
        return {"success": False, "reason": "Game is already over"}

    if isinstance(current_player, HumanPlayer):
        logger.info(f"Current player {current_player_id} is human. No AI move made.")
        return {"success": False, "reason": "Current player is human"}

    try:
        ai_action = current_player.select_action(state, game.legal_actions(state))
        new_state = game.next_state(state, ai_action)
        game_session["state"] = new_state
        logger.info(f"AI move made for player {current_player_id}. Action: {ai_action}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error making AI move: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making AI move")


@app.get("/{game_type}/{game_id}", response_class=HTMLResponse)
async def serve_game_page(request: Request, game_type: str, game_id: int):
    logger.debug(f"Serving game page. Type: {game_type}, ID: {game_id}")
    if game_type not in GAME_REGISTRY:
        logger.error(f"Invalid game type: {game_type}")
        raise HTTPException(status_code=404, detail="Game not found")
    if game_id not in games:
        logger.error(f"Game not found. ID: {game_id}")
        raise HTTPException(status_code=404, detail="Game not found")

    template_name = f"{game_type}.html"
    return templates.TemplateResponse(template_name, {"request": request, "game_type": game_type, "game_id": game_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

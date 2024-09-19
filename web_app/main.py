# web_app/main.py

import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rgi.core.game_registry import GAME_REGISTRY
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer
from typing import Dict, Any

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
    ai_player = data.get("ai_player", False)
    logger.info(f"Creating new game. Type: {game_type}, AI Player: {ai_player}")

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
        1: RandomPlayer(),
        2: MinimaxPlayer(game, player_id=2) if ai_player else RandomPlayer(),
    }
    logger.debug(f"Players initialized. Player 1: {type(players[1]).__name__}, Player 2: {type(players[2]).__name__}")

    # Store game session
    games[game_counter] = {
        "game": game,
        "serializer": game_serializer,
        "state": state,
        "players": players,
        "ai_player": ai_player,
    }
    logger.info(f"New game created. ID: {game_counter}, AI Player: {ai_player}")
    return {"game_id": game_counter, "game_type": game_type}


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
    game_state["ai_player"] = game_session["ai_player"]

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
    logger.debug(f"Making move for game ID: {game_id}. Action data: {action_data}")
    game_session = games.get(game_id)
    if not game_session:
        logger.error(f"Game not found. ID: {game_id}")
        raise HTTPException(status_code=404, detail="Game not found")

    game = game_session["game"]
    serializer = game_session["serializer"]
    state = game_session["state"]
    players = game_session["players"]
    ai_player = game_session["ai_player"]

    try:
        action = serializer.parse_action(game, action_data)
        if action not in game.legal_actions(state):
            logger.warning(f"Invalid move attempted. Game ID: {game_id}, Action: {action}")
            raise ValueError("Invalid move")
        state = game.next_state(state, action)
        logger.debug(f"Move made. New state: {state}")
    except ValueError as ve:
        logger.error(f"Error making move: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))

    # Update game state
    game_session["state"] = state

    # AI Move
    if ai_player and not game.is_terminal(state):
        current_player_id = game.current_player_id(state)
        ai = players.get(current_player_id)
        if isinstance(ai, MinimaxPlayer):
            logger.debug(f"AI (MinimaxPlayer) is making a move. Game ID: {game_id}")
            ai_action = ai.select_action(state, game.legal_actions(state))
            try:
                state = game.next_state(state, ai_action)
                game_session["state"] = state
                logger.debug(f"AI move made. New state: {state}")
            except ValueError as ve:
                logger.error(f"Error making AI move: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))

    return {"success": True}


@app.get("/{game_type}/{game_id}", response_class=HTMLResponse)
async def serve_game_page(request: Request, game_type: str, game_id: int):
    logger.debug(f"Serving game page. Type: {game_type}, ID: {game_id}")
    if game_type not in GAME_REGISTRY:
        logger.error(f"Invalid game type: {game_type}")
        raise HTTPException(status_code=404, detail="Game not found")
    if game_id not in games:
        logger.error(f"Game not found. ID: {game_id}")
        raise HTTPException(status_code=404, detail="Game not found")

    return templates.TemplateResponse("game.html", {"request": request, "game_type": game_type, "game_id": game_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

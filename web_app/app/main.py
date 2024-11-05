# web_app/main.py

import logging
from typing import Any, cast
from datetime import datetime
from threading import Lock

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rgi.core.game_registry import GAME_REGISTRY, PLAYER_REGISTRY, RegisteredGame
from rgi.players.human_player.human_player import HumanPlayer
from rgi.core.base import Game, Player, GameSerializer

print("Server restarted at", datetime.now())

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")


class ThreadSafeCounter:
    def __init__(self, initial_value: int = 0):
        self._value = initial_value
        self._lock = Lock()

    def increment(self) -> int:
        with self._lock:
            self._value += 1
            return self._value

    def current(self) -> int:
        with self._lock:
            return self._value


# In-memory storage for game sessions
GameSession = dict[str, Any]
games: dict[int, GameSession] = {}
game_counter = ThreadSafeCounter()


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/games/new")
async def create_game(request: Request) -> dict[str, Any]:
    data = await request.json()
    game_type: str = data.get("game_type", "")
    game_options: dict[str, Any] = data.get("game_options", {})
    player_options: dict[int, dict[str, Any]] = {int(k): v for k, v in data.get("player_options", {}).items()}
    logger.info(
        "Creating new game. Type: %s, Game Options: %s, Player Options: %s",
        game_type,
        game_options,
        player_options,
    )

    registry_entry = GAME_REGISTRY.get(game_type)
    if not registry_entry:
        logger.error("Invalid game type: %s", game_type)
        raise HTTPException(status_code=400, detail="Invalid game type")

    game = registry_entry.game_fn(**game_options)
    game_serializer = registry_entry.serializer_fn()
    state = game.initial_state()
    game_id = game_counter.increment()

    # Initialize players
    players: dict[int, Player[Any, Any, Any]] = {}
    for player_id, options in player_options.items():
        players[player_id] = create_player(
            options.get("player_type", "human"), game, registry_entry, player_id, options
        )
    logger.debug("Players initialized: %s", {pid: type(p).__name__ for pid, p in players.items()})

    # Store game session
    games[game_id] = {
        "game": game,
        "serializer": game_serializer,
        "state": state,
        "players": players,
        "game_options": game_options,
        "player_options": player_options,
    }
    logger.info(
        "New game created. ID: %d, Game Options: %s, Player Options: %s",
        game_id,
        game_options,
        player_options,
    )
    return {"game_id": game_id, "game_type": game_type}


def create_player(
    player_type: str,
    game: Game[Any, Any],
    registered_game: RegisteredGame[Any, Any],
    player_id: int,
    options: dict[str, Any],
) -> Player[Any, Any, Any]:
    # Remove 'player_type' from options to avoid passing it to the constructor
    constructor_options = {k: v for k, v in options.items() if k != "player_type"}

    player_fn = PLAYER_REGISTRY.get(player_type)
    if player_fn is None:
        logger.error("Unknown player type: %s", player_type)
        raise ValueError(f"Unknown player type: {player_type}")

    return player_fn(None, game, registered_game, player_id, constructor_options)


@app.get("/games/{game_id}/state")
async def get_game_state(game_id: int) -> dict[str, Any]:
    logger.debug("Fetching game state for game ID: %d", game_id)
    game_session = games.get(game_id)
    if not game_session:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    serializer = cast(GameSerializer[Game[Any, Any], Any, Any], game_session["serializer"])
    state = game_session["state"]
    game_state = serializer.serialize_state(game, state)

    # Add game and player options to the game state
    game_state["game_options"] = game_session["game_options"]
    game_state["player_options"] = game_session["player_options"]

    if game.is_terminal(state):
        all_players = range(1, game.num_players(state) + 1)
        rewards = {player_id: game.reward(state, player_id) for player_id in all_players}
        winner = [player_id for player_id, reward in rewards.items() if reward == 1.0]
        game_state["winner"] = winner[0] if winner else None
        logger.info("Game %d is terminal. Winner: %s", game_id, game_state["winner"])
    else:
        game_state["winner"] = None

    logger.debug("Game state for game %d: %s", game_id, game_state)
    return game_state


@app.post("/games/{game_id}/move")
async def make_move(game_id: int, action_data: dict[str, Any]) -> dict[str, Any]:
    game_session = games.get(game_id)
    if not game_session:
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    state = game_session["state"]

    try:
        serializer = cast(GameSerializer[Game[Any, Any], Any, Any], game_session["serializer"])
        action = serializer.parse_action(game, action_data)
        if action not in game.legal_actions(state):
            return {"success": False, "error": "Invalid move"}
        new_state = game.next_state(state, action)
        game_session["state"] = new_state
        return {"success": True}
    except ValueError as ve:
        return {"success": False, "error": str(ve)}


@app.post("/games/{game_id}/ai_move")
async def make_ai_move(game_id: int) -> dict[str, Any]:
    logger.debug("Attempting AI move for game ID: %d", game_id)
    game_session = games.get(game_id)
    if not game_session:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    game = cast(Game[Any, Any], game_session["game"])
    state = game_session["state"]
    players = cast(dict[int, Player[Any, Any, Any]], game_session["players"])

    current_player_id = game.current_player_id(state)
    current_player = players[current_player_id]

    if game.is_terminal(state):
        logger.info("Game %d is already in a terminal state.", game_id)
        return {"success": False, "reason": "Game is already over"}

    if isinstance(current_player, HumanPlayer):
        logger.info("Current player %d is human. No AI move made.", current_player_id)
        return {"success": False, "reason": "Current player is human"}

    try:
        legal_actions = game.legal_actions(state)
        ai_action = current_player.select_action(state, legal_actions)
        new_state = game.next_state(state, ai_action)
        game_session["state"] = new_state
        logger.info("AI move made for player %d. Action: %s", current_player_id, ai_action)
        return {"success": True}
    except Exception as e:
        logger.error("Error making AI move: %s", str(e))
        raise HTTPException(status_code=500, detail="Error making AI move") from e


@app.get("/{game_type}/{game_id}", response_class=HTMLResponse)
async def serve_game_page(request: Request, game_type: str, game_id: int) -> HTMLResponse:
    logger.debug("Serving game page. Type: %s, ID: %d", game_type, game_id)
    if game_type not in GAME_REGISTRY:
        logger.error("Invalid game type: %s", game_type)
        raise HTTPException(status_code=404, detail="Game not found")
    if game_id not in games:
        logger.error("Game not found. ID: %d", game_id)
        raise HTTPException(status_code=404, detail="Game not found")

    template_name = f"{game_type}.html"
    return templates.TemplateResponse(template_name, {"request": request, "game_type": game_type, "game_id": game_id})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

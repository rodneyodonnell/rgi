# web_app/main.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from rgi.games.connect4 import Connect4Game
from rgi.games.othello import OthelloGame
from typing import Dict, Any
from rgi.players.minimax_player import MinimaxPlayer
from rgi.players.random_player import RandomPlayer

app = FastAPI()
templates = Jinja2Templates(directory="web_app/templates")
app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# In-memory storage for game sessions
games: Dict[int, Dict[str, Any]] = {}
game_counter = 0


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Endpoint to create a new game
@app.post("/games/new")
async def create_game(request: Request):
    global game_counter
    data = await request.json()
    game_type = data.get("game_type")
    ai_player = data.get("ai_player", False)
    game_counter += 1
    if game_type == "connect4":
        game = Connect4Game()
        board_size = (game.height, game.width)  # (rows, columns)
    elif game_type == "othello":
        game = OthelloGame()
        board_size = (game.board_size, game.board_size)  # Assuming square board
    else:
        raise HTTPException(status_code=400, detail="Invalid game type")
    state = game.initial_state()
    # Store game, state, AI player info, and board size
    games[game_counter] = {"game": game, "state": state, "ai_player": ai_player, "board_size": board_size}
    return {"game_id": game_counter}


# Endpoint to get the game state
@app.get("/games/{game_id}/state")
async def get_game_state(game_id: int):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]["game"]
    state = games[game_id]["state"]
    board_size = games[game_id]["board_size"]

    # Serialize the board as a 2D list based on game type
    if isinstance(game, Connect4Game):
        rows, cols = board_size
        board_list = []
        for r in range(1, rows + 1):
            row = []
            for c in range(1, cols + 1):
                row.append(state.board.get((r, c), 0))
            board_list.append(row)
    elif isinstance(game, OthelloGame):
        size = board_size[0]  # Assuming square board
        board_list = []
        for r in range(1, size + 1):
            row = []
            for c in range(1, size + 1):
                row.append(state.board.get((r, c), 0))
            board_list.append(row)
    else:
        raise HTTPException(status_code=400, detail="Unknown game type")

    game_state = {
        "board": board_list,
        "current_player": state.current_player,
        "legal_actions": game.legal_actions(state),
        "is_terminal": game.is_terminal(state),
    }
    # Add winner information if the game is over
    if game_state["is_terminal"]:
        all_players = game.all_player_ids(state)
        rewards = {player_id: game.reward(state, player_id) for player_id in all_players}
        winner = [player_id for player_id, reward in rewards.items() if reward == 1.0]
        game_state["winner"] = winner[0] if winner else None
    else:
        game_state["winner"] = None
    return game_state


# Endpoint to make a move
@app.post("/games/{game_id}/move")
async def make_move(game_id: int, action: Dict[str, Any]):
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]["game"]
    state = games[game_id]["state"]
    ai_player = games[game_id]["ai_player"]
    board_size = games[game_id]["board_size"]
    current_player = game.current_player_id(state)

    # Human move
    if isinstance(game, Connect4Game):
        column = action.get("column")
        if column not in game.legal_actions(state):
            raise HTTPException(status_code=400, detail="Invalid move")
        state = game.next_state(state, column)
    elif isinstance(game, OthelloGame):
        row = action.get("row")
        col = action.get("col")
        position = (row, col)
        if position not in game.legal_actions(state):
            raise HTTPException(status_code=400, detail="Invalid move")
        state = game.next_state(state, position)
    else:
        raise HTTPException(status_code=400, detail="Invalid game type")

    games[game_id]["state"] = state

    # Check if AI should make a move
    if ai_player and not game.is_terminal(state):
        state = games[game_id]["state"]
        ai_player_id = game.current_player_id(state)
        ai = MinimaxPlayer(game, ai_player_id)
        ai_action = ai.select_action(state, game.legal_actions(state))
        if ai_action is not None:
            if isinstance(game, Connect4Game):
                ai_move = ai_action  # column number
            elif isinstance(game, OthelloGame):
                ai_move = {"row": ai_action[0], "col": ai_action[1]}
            state = game.next_state(state, ai_move)
            games[game_id]["state"] = state

    return {"success": True}


# Route to serve Connect Four game page
@app.get("/connect4/{game_id}", response_class=HTMLResponse)
async def connect4_game(request: Request, game_id: int):
    return templates.TemplateResponse("connect4.html", {"request": request, "game_id": game_id})


# Route to serve Othello game page
@app.get("/othello/{game_id}", response_class=HTMLResponse)
async def othello_game(request: Request, game_id: int):
    return templates.TemplateResponse("othello.html", {"request": request, "game_id": game_id})

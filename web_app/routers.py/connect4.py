# web_app/routers/connect4.py

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, Any
from rgi.games.connect4 import Connect4Game
from rgi.players.minimax_player import MinimaxPlayer

# Initialize router
router = APIRouter()

# Configure logger
logger = logging.getLogger(__name__)

# In-memory storage should be accessed from main.py or via a shared state
# For simplicity, we'll assume access to the 'games' dictionary via dependency injection or global access
# Here, we'll use a simplified approach

# Placeholder for game sessions (This should ideally be managed centrally)
# To maintain data consistency, consider using a shared state or a database
from web_app.main import games, serialize_board, get_winner, convert_ai_move


@router.get("/{game_id}", response_class=HTMLResponse)
async def connect4_game(request: Request, game_id: int):
    """
    Render the Connect Four game page.
    """
    if game_id not in games:
        logger.warning(f"Connect Four game page requested for non-existent Game ID {game_id}.")
        raise HTTPException(status_code=404, detail="Game not found")
    logger.info(f"Rendering Connect Four game page for Game ID {game_id}.")
    return templates.TemplateResponse("connect4.html", {"request": request, "game_id": game_id})


@router.get("/{game_id}/state", response_class=JSONResponse)
async def get_game_state(game_id: int):
    """
    Retrieve the current state of the Connect Four game.
    """
    if game_id not in games:
        logger.warning(f"Connect Four Game ID {game_id} not found.")
        raise HTTPException(status_code=404, detail="Game not found")
    game = games[game_id]["game"]
    state = games[game_id]["state"]
    board_size = games[game_id]["board_size"]

    logger.info(f"Fetching Connect Four state for Game ID: {game_id}")

    board_list = serialize_board(game, state, board_size)
    winner = get_winner(game, state)

    game_state = {
        "board": board_list,
        "current_player": state.current_player,
        "legal_actions": game.legal_actions(state),
        "is_terminal": game.is_terminal(state),
        "winner": winner,
    }

    logger.debug(f"Connect Four Game State for {game_id}: {game_state}")

    return game_state


@router.post("/{game_id}/move")
async def make_move(game_id: int, action: Dict[str, Any]):
    """
    Process a move in the Connect Four game.
    """
    if game_id not in games:
        logger.warning(f"Move attempted on non-existent Connect Four Game ID {game_id}.")
        raise HTTPException(status_code=404, detail="Game not found")

    game = games[game_id]["game"]
    state = games[game_id]["state"]
    ai_player = games[game_id]["ai_player"]
    board_size = games[game_id]["board_size"]
    current_player = game.current_player_id(state)

    logger.info(f"Connect Four Game {game_id}: Player {current_player} making move: {action}")

    try:
        column = action.get("column")
        if column is None or not isinstance(column, int):
            logger.error("Invalid or missing 'column' in move action for Connect Four.")
            raise HTTPException(status_code=400, detail="Invalid move format for Connect Four")
        if column not in game.legal_actions(state):
            logger.error(f"Invalid column move attempted in Connect Four: {column}")
            raise HTTPException(status_code=400, detail="Invalid move")
        state = game.next_state(state, column)
        games[game_id]["state"] = state
        logger.info(f"Connect Four Game {game_id}: Move processed successfully.")

        # AI move
        if ai_player and not game.is_terminal(state):
            logger.info(f"Connect Four Game {game_id}: AI is making a move.")
            ai_player_id = game.current_player_id(state)
            ai = MinimaxPlayer(game, ai_player_id)
            ai_action = ai.select_action(state, game.legal_actions(state))
            if ai_action is not None:
                ai_move = convert_ai_move(game, ai_action)
                logger.info(f"Connect Four Game {game_id}: AI selected move: {ai_move}")
                state = game.next_state(state, ai_move)
                games[game_id]["state"] = state
                logger.info(f"Connect Four Game {game_id}: AI move processed successfully.")

        return {"success": True}

    except ValueError as ve:
        logger.error(f"Connect Four Game {game_id}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Connect Four Game {game_id}: Unexpected error during move processing.")
        raise HTTPException(status_code=500, detail="Internal server error")

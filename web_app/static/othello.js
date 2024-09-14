// web_app/static/othello.js

function renderGame(data) {
    const gameDiv = document.getElementById('game');
    const statusDiv = document.getElementById('status');
    gameDiv.innerHTML = '';
    const table = document.createElement('table');
    table.classList.add('othello-table');
    const size = data.board.length;

    for (let r = size - 1; r >= 0; r--) {
        const row = document.createElement('tr');
        for (let c = 0; c < size; c++) {
            const cell = document.createElement('td');
            const value = data.board[r][c];
            if (value === 1) {
                cell.classList.add('othello-player1');
            } else if (value === 2) {
                cell.classList.add('othello-player2');
            } else {
                cell.classList.add('othello-empty');
            }

            // Add click handler if the move is legal
            if (data.legal_actions.some(action => action[0] === r + 1 && action[1] === c + 1)) {
                cell.onclick = () => makeMove({ row: r + 1, col: c + 1 });
            }

            row.appendChild(cell);
        }
        table.appendChild(row);
    }

    gameDiv.appendChild(table);

    if (data.is_terminal) {
        if (data.winner) {
            statusDiv.textContent = `Player ${data.winner} wins!`;
        } else {
            statusDiv.textContent = 'The game is a draw!';
        }
    } else {
        statusDiv.textContent = `Current Player: ${data.current_player}`;
    }
}

// Start the game
updateGameState();

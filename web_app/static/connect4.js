// web_app/static/connect4.js

function renderGame(data) {
    const gameDiv = document.getElementById('game');
    const statusDiv = document.getElementById('status');
    gameDiv.innerHTML = '';
    const table = document.createElement('table');
    table.classList.add('connect4-table');
    const rows = data.board.length;
    const cols = data.board[0].length;

    // Render rows from bottom to top
    for (let r = rows - 1; r >= 0; r--) {
        const row = document.createElement('tr');
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('td');
            const value = data.board[r][c];
            if (value === 1) {
                cell.classList.add('connect4-player1');
            } else if (value === 2) {
                cell.classList.add('connect4-player2');
            } else {
                cell.classList.add('connect4-empty');
            }

            // Add click handler if the move is legal
            if (data.legal_actions.includes(c + 1)) {
                cell.onclick = () => makeMove({ column: c + 1 });
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

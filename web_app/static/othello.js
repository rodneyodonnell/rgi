// web_app/static/othello.js

// Retrieve the game ID from the URL
const gameId = window.location.pathname.split("/").pop();

function updateGameState() {
    fetch(`/games/${gameId}/state`)
        .then(response => response.json())
        .then(data => {
            renderBoard(data.board, data.current_player, data.legal_actions, data.is_terminal, data.winner);
        })
        .catch(error => {
            console.error('Error fetching game state:', error);
            alert('Failed to fetch game state.');
        });
}

function renderBoard(boardData, currentPlayer, legalActions, isTerminal, winner) {
    const gameDiv = document.getElementById('game');
    const statusDiv = document.getElementById('status');
    gameDiv.innerHTML = '';
    const table = document.createElement('table');
    table.classList.add('othello-table');
    const size = boardData.length; // Assuming 8x8

    for (let r = 0; r < size; r++) {
        const row = document.createElement('tr');
        for (let c = 0; c < size; c++) {
            const cell = document.createElement('td');
            const value = boardData[r][c];
            if (value === 1) {
                cell.classList.add('othello-player1');
            } else if (value === 2) {
                cell.classList.add('othello-player2');
            } else {
                cell.classList.add('othello-empty');
            }

            // If this cell is a legal action, add onclick
            if (legalActions.some(action => action[0] === (r + 1) && action[1] === (c + 1))) {
                cell.onclick = () => makeMove(r + 1, c + 1); // rows and cols are 1-indexed
            }

            row.appendChild(cell);
        }
        table.appendChild(row);
    }

    gameDiv.appendChild(table);

    if (isTerminal) {
        if (winner) {
            statusDiv.textContent = `Player ${winner} wins!`;
        } else {
            statusDiv.textContent = 'The game is a draw!';
        }
    } else {
        statusDiv.textContent = `Current Player: ${currentPlayer}`;
    }
}

function makeMove(row, col) {
    console.log(`Attempting to make move at row: ${row}, column: ${col}`);
    fetch(`/games/${gameId}/move`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({row: row, col: col})
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('Move successful. Updating game state.');
            updateGameState();
        } else {
            console.log('Move failed. Invalid move.');
            alert('Invalid move');
        }
    })
    .catch(error => {
        console.error('Error making move:', error);
        alert('Failed to make move.');
    });
}

// Initialize the game by fetching the current state
updateGameState();

// Optional: Auto-refresh the game state every few seconds
// setInterval(updateGameState, 2000);

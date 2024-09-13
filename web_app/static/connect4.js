// web_app/static/connect4.js

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
    table.classList.add('connect4-table');
    const rows = boardData.length; // Typically 6
    const cols = boardData[0].length; // Typically 7

    // Render rows from bottom to top
    for (let r = rows - 1; r >= 0; r--) {
        const row = document.createElement('tr');
        for (let c = 0; c < cols; c++) {
            const cell = document.createElement('td');
            const value = boardData[r][c];
            if (value === 1) {
                cell.classList.add('connect4-player1');
            } else if (value === 2) {
                cell.classList.add('connect4-player2');
            } else {
                cell.classList.add('connect4-empty');
            }

            // If this column is a legal action, add onclick to the entire column
            if (legalActions.includes(c + 1)) {
                // Assign the column number as a data attribute
                cell.dataset.column = c + 1;
                cell.onclick = () => makeMove(c + 1);
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

function makeMove(column) {
    console.log(`Attempting to make move in column: ${column}`);
    fetch(`/games/${gameId}/move`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({column: column})
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

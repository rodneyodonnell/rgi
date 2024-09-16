// web_app/static/connect4.js

document.addEventListener('DOMContentLoaded', () => {
    console.log('connect4.js loaded and DOMContentLoaded event fired.');

    let isAI = false;
    let lastMove = null;
    let gameState = null;

    function startNewGame() {
        console.log('Starting a new Connect 4 game. Current AI setting:', isAI);

        fetch('/games/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                game_type: 'connect4', 
                ai_player: isAI 
            })
        })
        .then(response => {
            console.log('Response status:', response.status);
            if (!response.ok) {
                throw new Error(`Failed to create a new game. Status code: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('New game created with ID:', data.game_id);
            history.pushState(null, '', `/connect4/${data.game_id}`);
            
            const gameModal = bootstrap.Modal.getInstance(document.getElementById('gameModal'));
            if (gameModal) {
                gameModal.hide();
            }
            
            lastMove = null;
            gameState = null;
            
            updateGameState();
        })
        .catch(error => {
            console.error('Error creating new game:', error);
            showErrorToast('Failed to create a new game. Please try again.');
        });
    }

    window.renderGame = (data) => {
        console.log('Rendering game with data:', data);
        const gameArea = document.getElementById('game');
        const status = document.getElementById('status');
        const modalBody = document.getElementById('modalBody');
        const gameModal = new bootstrap.Modal(document.getElementById('gameModal'), {
            keyboard: false
        });
        const newGameButton = document.getElementById('newGameButton');

        isAI = data.ai_player;
        console.log('Updated AI setting:', isAI);

        gameArea.innerHTML = '';

        if (!data.rows || !data.columns) {
            console.error('Missing rows or columns in game state data.');
            gameArea.innerHTML = '<p>Error: Invalid game state data.</p>';
            showErrorToast('Invalid game state data received.');
            return;
        }

        const grid = document.createElement('div');
        grid.classList.add('grid-container');

        const reversedRows = [...data.state].reverse();

        for (let row = 0; row < reversedRows.length; row++) {
            for (let col = 0; col < data.columns; col++) {
                const cell = document.createElement('div');
                cell.classList.add('grid-cell');
                cell.dataset.column = col;
                cell.dataset.row = reversedRows.length - 1 - row;

                if (reversedRows[row][col] === 1) {
                    cell.classList.add('player1');
                } else if (reversedRows[row][col] === 2) {
                    cell.classList.add('player2');
                }

                if (lastMove && lastMove.row === (reversedRows.length - 1 - row) && lastMove.column === col) {
                    cell.classList.add('last-move');
                }

                if (!data.is_terminal) {
                    cell.addEventListener('click', () => {
                        console.log(`Cell clicked: Column ${col + 1}`);
                        makeMove({ column: col + 1 });
                    });
                } else {
                    cell.classList.add('inactive-cell');
                }

                grid.appendChild(cell);
            }
        }

        gameArea.appendChild(grid);

        if (data.is_terminal) {
            console.log('Game is terminal. Winner:', data.winner);
            let message = data.winner 
                ? `🎉 <strong>Player ${data.winner} Wins!</strong> 🎉`
                : '🤝 <strong>The game is a draw!</strong> 🤝';

            modalBody.innerHTML = message;
            gameModal.show();

            newGameButton.onclick = startNewGame;
        } else {
            console.log('Game continuing. Current player:', data.current_player);
            status.textContent = `Current Turn: Player ${data.current_player}`;
        }
    };

    function makeMove(action) {
        if (gameState && gameState.is_terminal) {
            console.log('Game is already over. Cannot make move.');
            return;
        }

        const gameId = window.location.pathname.split('/').pop();
        console.log(`Making move for game ${gameId}:`, action);

        fetch(`/games/${gameId}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(action)
        })
        .then(response => {
            console.log('Move response status:', response.status);
            if (!response.ok) {
                throw new Error('Failed to make move');
            }
            return response.json();
        })
        .then(data => {
            console.log('Move response:', data);
            if (data.success) {
                updateGameState(() => {
                    if (isAI && gameState && !gameState.is_terminal) {
                        setTimeout(() => {
                            console.log('Fetching AI move after delay');
                            updateGameState();
                        }, 300);
                    }
                });
            } else {
                showErrorToast('Invalid move. Please try again.');
            }
        })
        .catch(error => {
            console.error('Error making move:', error);
            showErrorToast('Failed to make move. Please try again.');
        });
    }

    function updateGameState(callback) {
        const gameId = window.location.pathname.split('/').pop();
        console.log('Updating game state for game ID:', gameId);

        fetch(`/games/${gameId}/state`)
            .then(response => {
                console.log('Game state response status:', response.status);
                if (!response.ok) {
                    throw new Error('Failed to fetch game state');
                }
                return response.json();
            })
            .then(data => {
                console.log('Received game state:', data);
                const newLastMove = findLastMove(data.state);
                if (newLastMove) {
                    lastMove = newLastMove;
                }
                gameState = data;
                renderGame(data);
                if (callback) callback();
            })
            .catch(error => {
                console.error('Error fetching game state:', error);
                showErrorToast('Failed to update game state.');
            });
    }

    function findLastMove(currentState) {
        if (!window.previousState) {
            window.previousState = currentState;
            return null;
        }

        for (let row = 0; row < currentState.length; row++) {
            for (let col = 0; col < currentState[row].length; col++) {
                if (currentState[row][col] !== window.previousState[row][col]) {
                    window.previousState = currentState;
                    return { row, column: col };
                }
            }
        }

        return null;
    }

    function showErrorToast(message) {
        const toastBody = document.getElementById('toastBody');
        if (!toastBody) {
            console.error('Toast body element not found.');
            return;
        }
        toastBody.textContent = message;

        const errorToastElement = document.getElementById('errorToast');
        if (!errorToastElement) {
            console.error('Error toast element not found.');
            return;
        }

        const errorToast = new bootstrap.Toast(errorToastElement, {
            delay: 5000
        });
        errorToast.show();
        console.log(`Error toast displayed: ${message}`);
    }

    updateGameState();
});
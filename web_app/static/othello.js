// web_app/static/othello.js

document.addEventListener('DOMContentLoaded', () => {
    console.log('othello.js loaded and DOMContentLoaded event fired.');

    // Define renderGame on window to make it globally accessible
    window.renderGame = (data) => {
        console.log('Rendering game with data:', data);
        const gameArea = document.getElementById('game');
        const status = document.getElementById('status');
        
        // Clear previous game board
        gameArea.innerHTML = '';

        // Check if rows and columns are present
        if (!data.rows || !data.columns) {
            console.error('Missing rows or columns in game state data.');
            gameArea.innerHTML = '<p>Error: Invalid game state data.</p>';
            return;
        }

        // Create game board grid
        const grid = document.createElement('div');
        grid.classList.add('grid-container');

        for (let row = 0; row < data.rows; row++) {
            for (let col = 0; col < data.columns; col++) {
                const cell = document.createElement('div');
                cell.classList.add('grid-cell');
                cell.dataset.row = row;
                cell.dataset.col = col;

                // Set cell color based on state
                if (data.state[row][col] === 1) {
                    cell.classList.add('player1');
                } else if (data.state[row][col] === 2) {
                    cell.classList.add('player2');
                }

                cell.addEventListener('click', () => {
                    console.log('Cell clicked:', { row: row, col: col });
                    makeMove({ row: row, col: col });
                });

                grid.appendChild(cell);
            }
        }

        gameArea.appendChild(grid);

        // Update game status
        if (data.is_terminal) {
            if (data.winner) {
                status.textContent = `Player ${data.winner} wins!`;
            } else {
                status.textContent = 'Game is a draw.';
            }
        } else {
            status.textContent = `Current Turn: Player ${data.current_player}`;
        }
    };

    // Initial game state fetch
    updateGameState();

    // Optionally, set an interval to periodically update the game state
    // setInterval(updateGameState, 5000);
});

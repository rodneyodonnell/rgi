import { BaseGameData, updateGameState, makeMove, showErrorToast } from './game_common.js';

interface OthelloGameData extends BaseGameData {
    rows: number;
    columns: number;
    state: number[][];
    legal_actions: [number, number][];
    options: {
        player1_type: string;
        player2_type: string;
    };
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('othello.ts loaded and DOMContentLoaded event fired.');

    const renderGame = (data: OthelloGameData) => {
        console.log('Rendering game with data:', JSON.stringify(data, null, 2));
        const gameArea = document.getElementById('game');
        const status = document.getElementById('status');
        
        if (!gameArea || !status) {
            console.error('Required DOM elements not found.');
            return;
        }

        // Clear previous game board
        gameArea.innerHTML = '';

        // Create game board grid
        const grid = document.createElement('div');
        grid.classList.add('grid-container');
        grid.style.gridTemplateColumns = `repeat(${data.columns}, 1fr)`;
        grid.style.gridTemplateRows = `repeat(${data.rows}, 1fr)`;

        console.log(`Creating grid with ${data.rows} rows and ${data.columns} columns`);

        for (let row = data.rows; row >= 1; row--) {
            for (let col = 1; col <= data.columns; col++) {
                const cell = document.createElement('div');
                cell.classList.add('grid-cell');
                cell.dataset.row = row.toString();
                cell.dataset.col = col.toString();

                console.log(`Creating cell at (${row}, ${col}) with state: ${data.state[row-1][col-1]}`);

                // Set cell color based on state
                if (data.state[row-1][col-1] === 1) {
                    cell.classList.add('player1');
                } else if (data.state[row-1][col-1] === 2) {
                    cell.classList.add('player2');
                }

                // Highlight legal moves
                if (data.legal_actions.some(([r, c]) => r === row && c === col)) {
                    cell.classList.add('legal-move');
                    console.log(`Legal move at (${row}, ${col})`);
                }

                cell.addEventListener('click', () => {
                    console.log('Cell clicked:', { row, col });
                    makeMove<OthelloGameData>({ row, col }, renderGame);
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

        console.log('Finished rendering game');
    };

    // Initial game state fetch
    console.log('Fetching initial game state');
    updateGameState<OthelloGameData>(renderGame);
});
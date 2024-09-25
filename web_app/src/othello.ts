import { BaseGameData, updateGameState, makeMove, showErrorToast, getCurrentGameId, startNewGame, makeAIMove, currentPlayerType } from './game_common.js';

interface OthelloGameData extends BaseGameData {
    rows: number;
    columns: number;
    state: number[][];
    legal_actions: [number, number][];
}

let gameOptions: { player1_type: string, player2_type: string };
let aiMoveInterval: number;

document.addEventListener('DOMContentLoaded', () => {
    console.log('othello.ts loaded and DOMContentLoaded event fired.');

    const renderGame = (data: OthelloGameData) => {
        console.log('Rendering game with data:', JSON.stringify(data, null, 2));
        const gameArea = document.getElementById('game');
        const status = document.getElementById('status');
        const modalBody = document.getElementById("modalBody");
        const gameModal = new window.bootstrap.Modal(document.getElementById("gameModal")!, {
            keyboard: false
        });
        const newGameButton = document.getElementById("newGameButton");
        
        if (!gameArea || !status) {
            console.error('Required DOM elements not found.');
            return;
        }

        gameOptions = data.options;
        console.log("Updated gameOptions:", gameOptions);

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

                if (!data.is_terminal && currentPlayerType(data) === "human") {
                    cell.addEventListener('click', () => {
                        console.log('Cell clicked:', { row, col });
                        makeMove<OthelloGameData>({ row, col }, renderGame);
                    });
                }

                grid.appendChild(cell);
            }
        }

        gameArea.appendChild(grid);

        // Update game status
        if (data.is_terminal) {
            console.log("Game is terminal. Winner:", data.winner);
            const message = data.winner
                ? `🎉 <strong>Player ${data.winner} Wins!</strong> 🎉`
                : "🤝 <strong>The game is a draw!</strong> 🤝";

            modalBody!.innerHTML = message;
            gameModal.show();  // Show the Bootstrap modal when game ends

            newGameButton!.onclick = () => {
                startNewGame("othello", gameOptions, renderGame)
                    .then(() => startAIMovePolling())
                    .catch(error => console.error("Error starting new game:", error));
            };
            stopAIMovePolling();
        } else {
            console.log("Game continuing. Current player:", data.current_player);
            status.textContent = `Current Turn: Player ${data.current_player}`;
        }

        console.log('Finished rendering game');
    };

    function startAIMovePolling() {
        stopAIMovePolling();  // Clear any existing interval
        aiMoveInterval = setInterval(() => makeAIMove(renderGame), 100);  // Poll every 100 ms
    }

    function stopAIMovePolling() {
        if (aiMoveInterval) {
            clearInterval(aiMoveInterval);
            aiMoveInterval = 0;
        }
    }

    // Initial game state fetch
    updateGameState(renderGame)
        .then(data => {
            gameOptions = data.options;
            startAIMovePolling();
        })
        .catch(error => console.error("Error fetching initial game state:", error));
});
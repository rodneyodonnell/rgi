import { BaseGameData, updateGameState, makeMove, showErrorToast, getCurrentGameId } from './game_common.js';

// No need to redeclare Window interface here, as it's already declared in game_common.ts

interface Connect4GameData extends BaseGameData {
    rows: number;
    columns: number;
    state: number[][];
    options: {
        player1_type: string;
        player2_type: string;
    };
}

let previousBoard: number[][];
let aiMoveInterval: number;
let gameOptions: { player1_type: string, player2_type: string };

document.addEventListener("DOMContentLoaded", () => {
    console.log("connect4.ts loaded and DOMContentLoaded event fired.");

    const renderGame = (data: Connect4GameData) => {
        console.log("Rendering game with data:", data);
        const gameArea = document.getElementById("game")!;
        const status = document.getElementById("status")!;
        const modalBody = document.getElementById("modalBody")!;
        const gameModal = new window.bootstrap.Modal(document.getElementById("gameModal")!, {
            keyboard: false
        });
        const newGameButton = document.getElementById("newGameButton");

        gameOptions = data.options;
        console.log("Updated gameOptions:", gameOptions);

        // Clear previous game board
        gameArea.innerHTML = "";

        // Create game board grid
        const grid = document.createElement("div");
        grid.classList.add("grid-container");

        for (let row = data.rows - 1; row >= 0; row--) {
            for (let col = 0; col < data.columns; col++) {
                const cell = document.createElement("div");
                cell.classList.add("grid-cell");
                cell.dataset.column = col.toString();
                cell.dataset.row = row.toString();

                if (data.state[row][col] === 1) {
                    cell.classList.add("player1");
                } else if (data.state[row][col] === 2) {
                    cell.classList.add("player2");
                }

                if (!data.is_terminal && currentPlayerType(data) === "human") {
                    cell.addEventListener("click", () => {
                        console.log(`Cell clicked: Column ${col}`);
                        makeMove<Connect4GameData>({ column: col + 1 }, renderGame);
                    });
                }

                grid.appendChild(cell);
            }
        }

        gameArea.appendChild(grid);

        // Compare the new board with the previous board to detect the last move
        if (data.state) {
            findLastMove(data.state);
            previousBoard = data.state;  // Update the previous board state
        }

        if (data.is_terminal) {
            console.log("Game is terminal. Winner:", data.winner);
            const message = data.winner
                ? `🎉 <strong>Player ${data.winner} Wins!</strong> 🎉`
                : "🤝 <strong>The game is a draw!</strong> 🤝";

            modalBody.innerHTML = message;
            gameModal.show();  // Show the Bootstrap modal when game ends

            newGameButton!.onclick = startNewGame;  // Ensure button triggers a new game
            stopAIMovePolling();
        } else {
            console.log("Game continuing. Current player:", data.current_player);
            status.textContent = `Current Turn: Player ${data.current_player}`;
        }
    };

    function startNewGame() {
        console.log("Starting a new Connect 4 game.");

        fetch("/games/new", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                game_type: "connect4",
                options: gameOptions
            })
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Failed to create a new game. Status code: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("New game created with ID:", data.game_id);
                // Update the URL with the new game ID
                window.history.pushState({}, "", `/connect4/${data.game_id}`);

                const gameModal = window.bootstrap.Modal.getInstance(document.getElementById("gameModal")!);
                if (gameModal) {
                    gameModal.hide();
                }

                updateGameState(renderGame);
                startAIMovePolling();
            })
            .catch(error => {
                console.error("Error creating new game:", error);
                showErrorToast("Failed to create a new game. Please try again.");
            });
    }

    function highlightLastMove(row: number, column: number) {
        // Remove the 'last-move' class from all previous moves
        document.querySelectorAll(".grid-cell.last-move").forEach(cell => {
            cell.classList.remove("last-move");
        });

        // Select the cell based on the row and column directly
        const lastMove = document.querySelector(`.grid-cell[data-column='${column}'][data-row='${row}']`);

        // Add the 'last-move' class to the selected cell
        if (lastMove) {
            lastMove.classList.add("last-move");
        }
    }

    function findLastMove(newBoard: number[][]) {
        if (!previousBoard) return;  // No previous board to compare to

        // Loop through the new board to find the difference
        for (let row = 0; row < newBoard.length; row++) {
            for (let col = 0; col < newBoard[row].length; col++) {
                // Check if there's a new piece (difference between old and new board)
                if (newBoard[row][col] != 0 && newBoard[row][col] !== previousBoard[row][col]) {
                    highlightLastMove(row, col);  // Highlight the last move
                    return;
                }
            }
        }
    }

    function currentPlayerType(data: Connect4GameData): string {
        return data.current_player === 1 ? data.options.player1_type : data.options.player2_type;
    }

    function makeAIMove() {
        const gameId = getCurrentGameId();
        console.log("Attempting AI move for game ID:", gameId);

        fetch(`/games/${gameId}/ai_move`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error("Failed to make AI move");
                }
                return response.json();
            })
            .then(data => {
                console.log("AI move response:", data);
                if (data.success) {
                    updateGameState(renderGame);
                }
            })
            .catch(error => {
                console.error("Error making AI move:", error);
            });
    }

    function startAIMovePolling() {
        stopAIMovePolling();  // Clear any existing interval
        aiMoveInterval = setInterval(makeAIMove, 100);  // Poll every 100 ms
    }

    function stopAIMovePolling() {
        if (aiMoveInterval) {
            clearInterval(aiMoveInterval);
            aiMoveInterval = 0;
        }
    }

    // Start AI move polling when the page loads
    startAIMovePolling();

    // Initial game state fetch
    updateGameState(renderGame);
});
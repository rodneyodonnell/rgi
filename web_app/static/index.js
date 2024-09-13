// web_app/static/index.js

function startGame(gameType, aiPlayer) {
    fetch('/games/new', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({game_type: gameType, ai_player: aiPlayer})
    })
    .then(response => response.json())
    .then(data => {
        const gameId = data.game_id;
        if (gameType === 'connect4') {
            window.location.href = `/connect4/${gameId}`;
        } else if (gameType === 'othello') {
            window.location.href = `/othello/${gameId}`;
        }
    })
    .catch(error => {
        console.error('Error creating game:', error);
        alert('Failed to create game. Please try again.');
    });
}

// web_app/static/game_common.js

function updateGameState() {
    fetch(`/games/${GAME_ID}/state`)
        .then(response => response.json())
        .then(data => {
            if (window.renderGame) {
                window.renderGame(data);
            } else {
                console.error('renderGame function is not defined');
            }
        })
        .catch(error => {
            console.error('Error fetching game state:', error);
            alert('Failed to fetch game state.');
        });
}

function makeMove(action) {
    fetch(`/games/${GAME_ID}/move`, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(action)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            updateGameState();
        } else {
            alert('Invalid move');
        }
    })
    .catch(error => {
        console.error('Error making move:', error);
        alert('Failed to make move.');
    });
}

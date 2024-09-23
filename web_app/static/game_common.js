// web_app/static/game_common.js
"use strict";

function updateGameState() {
    console.log(`Fetching game state for GAME_ID: ${GAME_ID}`);
    fetch(`/games/${GAME_ID}/state`)
        .then(response => {
            if (!response.ok) {
                console.error(`HTTP error! Status: ${response.status}`);
                throw new Error(`Failed to fetch game state. Status code: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Game state data received:', data);
            if (window.renderGame) {
                window.renderGame(data);
            } else {
                console.error('renderGame function is not defined');
            }
        })
        .catch(error => {
            console.error('Error fetching game state:', error);
            showErrorToast('Failed to fetch game state.');
        });
}

function makeMove(action) {
    console.log(`Making move:`, action);
    fetch(`/games/${GAME_ID}/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(action)
    })
    .then(response => {
        if (!response.ok) {
            // Extract error message from the response
            return response.json().then(errData => {
                const errorMsg = errData.detail || 'Failed to make move.';
                console.error(`Move failed: ${errorMsg}`);
                throw new Error(errorMsg);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Move response:', data);
        if (data.success) {
            updateGameState();
        } else {
            console.warn('Move was not successful:', data);
            showErrorToast('Invalid move. Please try again.');
        }
    })
    .catch(error => {
        console.error('Error making move:', error);
        showErrorToast(error.message || 'Failed to make move.');
    });
}

// Function to show error toast
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

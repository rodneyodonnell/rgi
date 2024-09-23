// Import the types we need from bootstrap
import { Toast, Modal } from 'bootstrap';

// Declare the specific Bootstrap components we're using
declare global {
    interface Window {
        bootstrap: {
            Toast: typeof Toast;
            Modal: typeof Modal & {
                getInstance: (element: Element | null) => Modal | null;
            };
        }
    }
}

export interface BaseGameData {
    is_terminal: boolean;
    winner: number | null;
    current_player: number;
}

export function getCurrentGameId(): string {
    return window.location.pathname.split("/").pop() || "";
}

export function showErrorToast(message: string) {
    const toastBody = document.getElementById("toastBody");
    if (!toastBody) {
        console.error("Toast body element not found.");
        return;
    }
    toastBody.textContent = message;

    const errorToastElement = document.getElementById("errorToast");
    if (!errorToastElement) {
        console.error("Error toast element not found.");
        return;
    }

    const errorToast = new window.bootstrap.Toast(errorToastElement);
    errorToast.show();
    console.log(`Error toast displayed: ${message}`);
}

export function updateGameState<T extends BaseGameData>(renderGame: (data: T) => void) {
    const gameId = getCurrentGameId();
    console.log("Updating game state for game ID:", gameId);

    fetch(`/games/${gameId}/state`)
        .then(response => {
            if (!response.ok) {
                throw new Error("Failed to fetch game state");
            }
            return response.json();
        })
        .then(data => {
            console.log("Received game state:", data);
            renderGame(data as T);
        })
        .catch(error => {
            console.error("Error fetching game state:", error);
            showErrorToast("Failed to fetch game state.");
        });
}

export function makeMove<T extends BaseGameData>(
    action: { [key: string]: number | string },
    renderGame: (data: T) => void
) {
    const gameId = getCurrentGameId();
    console.log(`Making move for game ${gameId}:`, action);
    fetch(`/games/${gameId}/move`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(action)
    })
    .then(response => response.json())
    .then(data => {
        console.log("Move response:", data);
        if (data.success) {
            updateGameState(renderGame);
        } else {
            showErrorToast(data.error || "Unknown error occurred");
        }
    })
    .catch(error => {
        console.error("Error making move:", error);
        showErrorToast(error.message || "Failed to make move.");
    });
}
import {
    BaseGameData,
    updateGameState,
    makeMove,
    startNewGame,
    currentPlayerType,
} from './game_common.js'

interface Infiltr8State extends BaseGameData {
    deck_size: number;
    discard_pile: { name: string; value: number; description: string }[];
    players: {
        [key: number]: {
            is_protected: boolean;
            is_out: boolean;
            hand: { name: string; description: string }[];
        };
    };
    current_player: number;
    legal_actions: Infiltr8Action[];
    game_options: { [key: string]: any };
    player_options: { [key: number]: { player_type: string; [key: string]: any } };
    action_log: string[];
}

interface Infiltr8Action {
    action_type: "DRAW" | "PLAY";
    card?: string;
    player_id?: number;
    guess_card?: string;
}

let gameOptions: { [key: string]: any } = {};
let playerOptions: { [key: number]: { player_type: string; [key: string]: any } } = {};

document.addEventListener('DOMContentLoaded', () => {
    console.log('infiltr8.ts loaded and DOMContentLoaded event fired.')

    const renderGame = (data: Infiltr8State) => {
        console.log('Rendering game with data:', JSON.stringify(data, null, 2))
        const gameArea = document.querySelector('.game-area')
        const status = document.getElementById('status')
        const actionForm = document.getElementById('action-form')
        const actionLogList = document.getElementById('action-log-list')

        if (!gameArea || !status || !actionForm || !actionLogList) {
            console.error('Required DOM elements not found.')
            return
        }

        let html = ''

        // Render players
        for (const [id, player] of Object.entries(data.players)) {
            const isCurrentPlayer = parseInt(id) === data.current_player
            const playerStatus = player.is_out ? 'eliminated' : player.is_protected ? 'protected' : 'active'
            html += `
                <div class="player-area ${playerStatus} ${isCurrentPlayer ? 'current-player' : ''}">
                    <div class="player-header">
                        <h3>Player ${id}</h3>
                        ${isCurrentPlayer ? '<div class="current-player-indicator" id="player-indicator-${id}"></div>' : ''}
                    </div>
                    <p class="player-status ${playerStatus}">${playerStatus.charAt(0).toUpperCase() + playerStatus.slice(1)}</p>
                    <div class="hand">
                        ${player.hand.map(card => `
                            <div class="card" title="${card.description}">
                                <div class="card-body">
                                    <h5 class="card-title">${card.name}</h5>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `
        }

        // Render deck and discard pile
        html += `
            <div class="deck-and-discard">
                <div class="deck">
                    <h3>Deck</h3>
                    <div class="card-stack">
                        <div class="card-back"></div>
                        <p class="cards-remaining">${data.deck_size}</p>
                    </div>
                </div>
                <div class="discard-pile">
                    <h3>Discard Pile</h3>
                    <div class="discard-stack">
                        ${data.discard_pile.map((card, index) => `
                            <div class="card discard-card" style="top: ${index * 5}px;" title="${card.description}">
                                <div class="card-body">
                                    <h5 class="card-title">${card.name}</h5>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `

        gameArea.innerHTML = html

        // Update status
        status.textContent = `Current Turn: Player ${data.current_player}`

        // Render action form
        actionForm.innerHTML = renderActionForm(data)

        // Render action log
        actionLogList.innerHTML = data.action_log && data.action_log.length > 0
            ? data.action_log.map(action => `<li class="list-group-item">${action}</li>`).join('')
            : '<li class="list-group-item">No actions yet.</li>'

        console.log('Finished rendering game')
    }

    function renderActionForm(data: Infiltr8State) {
        if (data.is_terminal || currentPlayerType(data) !== 'human') {
            return ''
        }

        let html = '<select class="form-select action-select">'
        for (const action of data.legal_actions) {
            html += `<option value='${JSON.stringify(action)}'>${renderAction(action, data.current_player)}</option>`
        }
        html += '</select>'
        html += '<button class="btn btn-primary submit-action mt-2">Submit Action</button>'

        return html
    }

    function renderAction(action: Infiltr8Action, currentPlayer: number): string {
        if (action.action_type === "DRAW") {
            return "Draw a card"
        } else if (action.action_type === "PLAY") {
            let actionText = `Player ${currentPlayer} played ${action.card}`
            if (action.player_id !== undefined && action.player_id !== null) {
                actionText += ` on Player ${action.player_id}`
            }
            if (action.card === "Hack" && action.guess_card !== undefined) {
                actionText += ` guessing ${action.guess_card}`
            }
            return actionText
        }
        return JSON.stringify(action)
    }

    // Initial game state fetch
    updateGameState(renderGame)
        .then((data) => {
            gameOptions = data.game_options
            playerOptions = data.player_options
        })
        .catch((error) =>
            console.error('Error fetching initial game state:', error),
        )

    // Set up event listener for action form
    document.getElementById('action-form')?.addEventListener('click', (e) => {
        const target = e.target as HTMLElement
        if (target.classList.contains('submit-action')) {
            const select = document.querySelector('.action-select') as HTMLSelectElement
            const action = JSON.parse(select.value)
            makeMove(action, renderGame)
        }
    })

    // Set up new game button
    document.getElementById('newGameButton')?.addEventListener('click', () => {
        startNewGame('infiltr8', gameOptions, playerOptions, renderGame)
    })
})
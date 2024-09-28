import {
    BaseGameData,
    updateGameState,
    makeMove,
    startNewGame,
    currentPlayerType,
} from './game_common.js'

interface Infiltr8State extends BaseGameData {
    deck_size: number;
    discard_pile: { name: string; value: number }[];
    players: {
        [key: number]: {
            is_protected: boolean;
            is_out: boolean;
            hand: string[];
        };
    };
    current_player: number;
    legal_actions: Infiltr8Action[];
}

interface Infiltr8Action {
    action_type: "DRAW" | "PLAY";
    card?: string;
    player_id?: number;
    guess_card?: string;
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('infiltr8.ts loaded and DOMContentLoaded event fired.')

    const renderGame = (data: Infiltr8State) => {
        console.log('Rendering game with data:', JSON.stringify(data, null, 2))
        const gameArea = document.getElementById('game')
        const status = document.getElementById('status')
        const actionForm = document.getElementById('action-form')

        if (!gameArea || !status || !actionForm) {
            console.error('Required DOM elements not found.')
            return
        }

        let html = ''

        // Render players
        for (const [id, player] of Object.entries(data.players)) {
            const isCurrentPlayer = parseInt(id) === data.current_player
            html += `
                <div class="player-area ${isCurrentPlayer ? 'current-player' : ''}">
                    <h3>Player ${id} ${isCurrentPlayer ? '(Current Player)' : ''}</h3>
                    <p>Status: ${player.is_protected ? 'Protected' : player.is_out ? 'Out' : 'Active'}</p>
                    <p>Hand: ${player.hand.join(', ')}</p>
                </div>
            `
        }

        // Render deck and discard pile
        html += `
            <div class="deck">
                <h3>Deck</h3>
                <p>Cards remaining: ${data.deck_size}</p>
            </div>
            <div class="discard-pile">
                <h3>Discard Pile</h3>
                ${data.discard_pile.map(card => `<div class="card">${card.name}</div>`).join('')}
            </div>
        `

        gameArea.innerHTML = html

        // Update status
        status.textContent = `Current Turn: Player ${data.current_player}`

        // Render action form
        actionForm.innerHTML = renderActionForm(data)

        console.log('Finished rendering game')
    }

    function renderActionForm(data: Infiltr8State) {
        if (data.is_terminal || currentPlayerType(data) !== 'human') {
            return ''
        }

        let html = '<select class="action-select">'
        for (const action of data.legal_actions) {
            html += `<option value='${JSON.stringify(action)}'>${renderAction(action)}</option>`
        }
        html += '</select>'
        html += '<button class="submit-action mt-2">Submit Action</button>'

        return html
    }

    function renderAction(action: Infiltr8Action): string {
        if (action.action_type === "DRAW") {
            return "Draw a card"
        } else if (action.action_type === "PLAY") {
            let actionText = `Play ${action.card}`
            if (action.player_id !== undefined) {
                actionText += ` on Player ${action.player_id}`
            }
            if (action.guess_card !== undefined) {
                actionText += ` guessing ${action.guess_card}`
            }
            return actionText
        }
        return JSON.stringify(action)
    }

    // Initial game state fetch
    updateGameState(renderGame)

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
        startNewGame('infiltr8', { player1_type: 'human', player2_type: 'ai' }, renderGame)
    })
})
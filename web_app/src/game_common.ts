import { Toast, Modal } from 'bootstrap'

const urlParams = new URLSearchParams(window.location.search)
const aiIntervalMs = Number(urlParams.get('ai_interval_ms') ?? '100')

declare global {
  interface Window {
    bootstrap: {
      Toast: typeof Toast
      Modal: typeof Modal & {
        getInstance: (element: Element | null) => Modal | null
      }
    }
  }
}

export interface BaseGameData {
  is_terminal: boolean
  winner: number | null
  current_player: number
  options: {
    player1_type: string
    player2_type: string
  }
}

export function getCurrentGameId(): string {
  return window.location.pathname.split('/').pop() || ''
}

export function showErrorToast(message: string) {
  const toastBody = document.getElementById('toastBody')
  if (!toastBody) {
    console.error('Toast body element not found.')
    return
  }
  toastBody.textContent = message

  const errorToastElement = document.getElementById('errorToast')
  if (!errorToastElement) {
    console.error('Error toast element not found.')
    return
  }

  const errorToast = new window.bootstrap.Toast(errorToastElement)
  errorToast.show()
  console.log(`Error toast displayed: ${message}`)
}

export function updateGameState<T extends BaseGameData>(
  renderGame: (data: T) => void,
): Promise<T> {
  const gameId = getCurrentGameId()
  console.log('Updating game state for game ID:', gameId)

  return fetch(`/games/${gameId}/state`)
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to fetch game state')
      }
      return response.json()
    })
    .then((data) => {
      console.log('Received game state:', data)
      renderGame(data as T)
      return data as T
    })
    .catch((error) => {
      console.error('Error fetching game state:', error)
      showErrorToast('Failed to fetch game state.')
      throw error
    })
}

export function makeMove<T extends BaseGameData>(
  action: { [key: string]: number | string },
  renderGame: (data: T) => void,
) {
  const gameId = getCurrentGameId()
  console.log(`Making move for game ${gameId}:`, action)
  fetch(`/games/${gameId}/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log('Move response:', data)
      if (data.success) {
        updateGameState(renderGame)
      } else {
        showErrorToast(data.error || 'Unknown error occurred')
      }
    })
    .catch((error) => {
      console.error('Error making move:', error)
      showErrorToast(error.message || 'Failed to make move.')
    })
}

export function startNewGame(
  gameType: string,
  gameOptions: { player1_type: string; player2_type: string },
  renderGame: (data: any) => void,
): Promise<void> {
  console.log(`Starting a new ${gameType} game with options:`, gameOptions)

  return fetch('/games/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_type: gameType,
      options: gameOptions,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(
          `Failed to create a new game. Status code: ${response.status}`,
        )
      }
      return response.json()
    })
    .then((data) => {
      console.log('New game created with ID:', data.game_id)
      // Update the URL with the new game ID
      window.history.pushState(
        {},
        '',
        `/${gameType}/${data.game_id}${window.location.search}`,
      )

      const gameModal = window.bootstrap.Modal.getInstance(
        document.getElementById('gameModal')!,
      )
      if (gameModal) {
        gameModal.hide()
      }

      return updateGameState(renderGame)
    })
    .catch((error) => {
      console.error('Error creating new game:', error)
      showErrorToast('Failed to create a new game. Please try again.')
      throw error
    })
}

export function makeAIMove(renderGame: (data: any) => void) {
  const gameId = getCurrentGameId()
  console.log('Attempting AI move for game ID:', gameId)

  fetch(`/games/${gameId}/ai_move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error('Failed to make AI move')
      }
      return response.json()
    })
    .then((data) => {
      console.log('AI move response:', data)
      if (data.success) {
        updateGameState(renderGame)
      }
    })
    .catch((error) => {
      console.error('Error making AI move:', error)
    })
}

export function currentPlayerType(data: BaseGameData): string {
  return data.current_player === 1
    ? data.options.player1_type
    : data.options.player2_type
}

let aiMoveInterval: number

export function startAIMovePolling(renderGame: (data: any) => void) {
  stopAIMovePolling() // Clear any existing interval
  aiMoveInterval = setInterval(() => makeAIMove(renderGame), aiIntervalMs) // Poll every 100 ms
}

export function stopAIMovePolling() {
  if (aiMoveInterval) {
    clearInterval(aiMoveInterval)
    aiMoveInterval = 0
  }
}

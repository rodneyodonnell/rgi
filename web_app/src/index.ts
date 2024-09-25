// web_app/static/index.js

document.addEventListener('DOMContentLoaded', () => {
  const connect4Form = document.getElementById('connect4Form')
  const othelloForm = document.getElementById('othelloForm')

  connect4Form?.addEventListener('submit', (e) => {
    e.preventDefault()
    const options = {
      player1_type: (
        document.getElementById('connect4Player1') as HTMLInputElement
      )?.value,
      player2_type: (
        document.getElementById('connect4Player2') as HTMLInputElement
      )?.value,
    }
    startGame('connect4', options)
  })

  othelloForm?.addEventListener('submit', (e) => {
    e.preventDefault()
    const options = {
      player1_type: (
        document.getElementById('othelloPlayer1') as HTMLInputElement
      )?.value,
      player2_type: (
        document.getElementById('othelloPlayer2') as HTMLInputElement
      )?.value,
    }
    startGame('othello', options)
  })
})

function startGame(
  gameType: string,
  options: { player1_type: string; player2_type: string },
) {
  console.log('Starting game:', { gameType, options })
  fetch('/games/new', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      game_type: gameType,
      options: options,
    }),
  })
    .then((response) => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`)
      }
      return response.json()
    })
    .then((data) => {
      console.log('Game created with ID:', data.game_id)
      window.location.href = `/${gameType}/${data.game_id}${window.location.search}`
    })
    .catch((error) => {
      console.error('Error creating game:', error)
      alert('Failed to create game. Please try again.')
    })
}

# This file contains extra context to be used in Cursor and by any bots.

## Project Overview
You are helping me build a game playing AI we're calling RGI.
Am an an experienced engineer with lots of ML & backend experience but little frontend.
So please explain non-obvious frontend concepts.

The planned approach is to:
- Implement several games in python, initially simple games like connect4 and expanding to euro games like power grid.
- Implement several algorithms, starting with minimax and expanding to more complex like alphazero/muzero/gato.
- Implement a frontend for humans to play games against bots.
- Implement a frontend for humans to play games against each other.
- Genralize the games & algorithms as much as possible in the game domain.
- Use transfer learning to efficiently produce models of new games/
- Eventually expand to real world tasks.

Bot instructions
- Please keep replies terse, and never appologise.
- Please call out bullshit ideas, and don't always be positive.
- Please ask for clarification when needed.
- Please think through ideas step by step before committing to an approach.
- Please update this file whenever the file or class structure is changes.
- Please add any context you would fine useful in the future to this file.
- Please ask for extra context when I have not supplied enough information.
- Please add any notes to yourself in the appropriate section at the bottom of this file.

Tech stack:
- Linux (ubuntu)
- Python 3.11 (backend)
- Python FastAPI (backend API)
- typescript (frontend)
- Jax (training & ML)
- Docker (containerization)


## TODO:

This is a list of tasks we are working on now or planning to work on soon.

- Infiltr8: Show log of actions.
- Infiltr8: Update player state based on actions (peek, etc.)
- Infiltr8: Protection should not stop a player being targeted. So you are not forced to play superintelligence if all players are protected.
- Infiltr8: Should only show things we know.
- Infiltr8: Handle case where no actionsa are legal (opponents all protected)
- Expand games: not board
- Expand games: not sequential turns
- Split cursor_context.md into separte design docs for each game, algorithm, etc.


## Python Style Guide.

Please follow these rules for all pyton code.
- Use modern python features from Python 3.11
- Include types in python code where appropriate.
- Never import `List` or `Tuple` from `typing`. Always use the more modern `list` and `tuple` (lowercase) instead.
- When creating a TypeVar, prefix the name with T. E.g. `TState = TypeVar('TState')`
- Use `@override` decorator when overriding methods (`from typing_extensions import override`).
- I'm using black to autoformat the code, so use `# fmt: off` and `# fmt: on` to disable formatting where needed.
- Follow existing conventions in the code.
- Aim for simplicity and clarity where possible.

## Typescript Style Guide.

- Follow the conventions in the existing code.
- Aim for simplicity and clarity where possible.

## Project Conventions:
When working with games, follow the following conentions unless there is a strong convention for the particular game:
- Always use (row, col) for coordinates, with row before column.
- In games with boards, use (1, 1) for the bottom-left corner.




## Python Class Design

- Game & Action do not store any state and they are never modified after cration.
- TGameState and TPlayerState are immutable. They should usually be based on dataclasses and python's `immutables` library for performance.

```python
from abc import ABC
from typing import Generic, TypeVar, Any

TGame = TypeVar("TGame", bound="Game[Any, Any, Any]")
TGameState = TypeVar("TGameState")
TPlayerState = TypeVar("TPlayerState")
TPlayerId = TypeVar("TPlayerId")
TAction = TypeVar("TAction")
TEmbedding = TypeVar("TEmbedding")

class Game(ABC, Generic[TGameState, TPlayerId, TAction]):
    def initial_state(self) -> TGameState:
    def current_player_id(self, state: TGameState) -> TPlayerId:
    def all_player_ids(self, state: TGameState) -> list[TPlayerId]:
    def legal_actions(self, state: TGameState) -> list[TAction]:
    def next_state(self, state: TGameState, action: TAction) -> TGameState:
    def is_terminal(self, state: TGameState) -> bool:
    def reward(self, state: TGameState, player_id: TPlayerId) -> float:
    def pretty_str(self, state: TGameState) -> str:

class StateEmbedder(ABC, Generic[TGameState, TEmbedding]):
    def embed_state(self, state: TGameState) -> TEmbedding:
    def get_embedding_dim(self) -> int:

class ActionEmbedder(ABC, Generic[TAction, TEmbedding]):
    def embed_action(self, action: TAction) -> TEmbedding:
    def get_embedding_dim(self) -> int:

class GameSerializer(ABC, Generic[TGame, TGameState, TAction]):
    def serialize_state(self, game: TGame, state: TGameState) -> dict[str, Any]:
    def parse_action(self, game: TGame, action_data: dict[str, Any]) -> TAction:

class Player(ABC, Generic[TGameState, TPlayerState, TAction]):
    def select_action(self, game_state: TGameState, legal_actions: list[TAction]) -> TAction:
    def update_state(self, game_state: TGameState, action: TAction) -> None:

class GameObserver(ABC, Generic[TGameState, TPlayerId, TAction]):
    def observe_initial_state(self, state: TGameState) -> None:
    def observe_action(self, state: TGameState, player: TPlayerId, action: TAction) -> None:
    def observe_state_transition(self, old_state: TGameState, new_state: TGameState) -> None:
    def observe_game_end(self, final_state: TGameState) -> None:
```

## TypeScript Interfaces and Functions


```typescript
interface BaseGameData {
    is_terminal: boolean;
    winner: number | null;
    current_player: number;
    game_options: { [key: string]: any };
    player_options: { [key: number]: { player_type: string; [key: string]: any } };
}

function updateGameState<T extends BaseGameData>(renderGame: (data: T) => void): Promise<T>;

function makeMove<T extends BaseGameData>(
    action: { [key: string]: number | string },
    renderGame: (data: T) => void
): void;

function startNewGame(
    gameType: string,
    gameOptions: { [key: string]: any },
    playerOptions: { [key: number]: { player_type: string; [key: string]: any } },
    renderGame: (data: any) => void
): Promise<void>;

function makeAIMove(renderGame: (data: any) => void): void;

function currentPlayerType(data: BaseGameData): string;
```

## File Structure
```bash
.
|-- bot_artifacts
|-- logs
|-- notebooks
|-- pdfs
|-- rgi
|   |-- config
|   |-- core
|   |   |-- base.py
|   |   |-- game_registry.py
|   |   `-- __init__.py
|   |-- games
|   |   |-- game1.py
|   |   |-- game2.py
|   |   `-- __init__.py
|   |-- models
|   |   `-- __init__.py
|   |-- players
|   |   |-- human_player.py
|   |   |-- __init__.py
|   |   |-- minimax_player.py
|   |   `-- random_player.py
|   |-- tests
|   |   |-- games
|   |   |-- players
|   |   `-- __init__.py
|   |-- __init__.py
|   `-- main.py
|-- scripts
|-- web_app
|   |-- app
|   |   `-- main.py
|   |-- src
|   |   |-- game1.ts
|   |   |-- game2.ts
|   |   |-- game_common.ts
|   |   `-- index.ts
|   |-- static
|   |   |-- game1.js
|   |   |-- game2.js
|   |   |-- game_common.js
|   |   |-- index.js
|   |   |-- favicon.ico
|   |   |-- rgi_512x512.png
|   |   `-- styles.css
|   |-- styles
|   |-- templates
|   |   |-- base.html
|   |   |-- game1.html
|   |   |-- game2.html
|   |   |-- game.html
|   |   `-- index.html
|   `-- tests
|       |-- test_game1_frontend.py
|       |-- test_game2_frontend.py
|       `-- test_homepage.py
|-- Dockerfile
|-- eslint.config.mjs
|-- package.json
|-- pyproject.toml
|-- README.md
|-- requirements.in
|-- requirements.txt
|-- tsconfig.json
`-- yarn.lock
```

## Development Environment and Workflow

Note: This section was written by Claude, the AI assistant, primarily as a scratchpad for longer-term thoughts and observations. Claude should feel free to update this section at any time in the future as needed.

- Docker-based development with GPU support (TensorFlow 2.17.0)
- VSCode as primary IDE with extensions for Python, TypeScript, and Docker
- FastAPI backend with Uvicorn server
- TypeScript frontend
- Playwright for frontend testing
- pytest for Python testing
- mypy and pylint for Python linting
- ESLint and TypeScript compiler for TypeScript linting
- Jupyter notebooks for experimentation

Key commands:
- Build and run Docker: `docker build -t rgi-gpu . && docker run -it --gpus all -v $(pwd)/logs:/app/logs rgi-gpu`
- Run tests: `pytest`
- Reformat code: `black . --line-length 120`
- Run frontend tests: `pytest web_app/tests/test_connect4_frontend.py`
- Auto-build TypeScript: `yarn tsc --watch`

VSCode Configuration:
- Launch configuration for FastAPI:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "web_app.app.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                "8000",
                "--reload"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
}
```

- Dev Container configuration:
```json
{
    "name": "RGI Dev Container",
    "build": {
        "context": "..",
        "dockerfile": "../Dockerfile",
        "args": {
            "USERNAME": "vscode"
        }
    },
    "forwardPorts": [8888],
    "runArgs": ["--gpus", "all"],
    "mounts": [
        "source=${localWorkspaceFolder}/logs,target=/app/logs,type=bind",
        "source=${localWorkspaceFolder}/rgi,target=/app/rgi,type=bind",
        "source=${localWorkspaceFolder}/web_app,target=/app/web_app,type=bind",
        "source=${localWorkspaceFolder}/scripts,target=/app/scripts,type=bind",
        "source=${localWorkspaceFolder}/notebooks,target=/app/notebooks,type=bind"
    ],
    "remoteUser": "vscode",
    "updateRemoteUserUID": true,
    "customizations": {
        "vscode": {
            "extensions": [
                "GitHub.copilot",
                "ms-python.python",
                "ms-vscode.git",
                "ms-azuretools.vscode-docker",
                "ms-toolsai.jupyter",
                "ms-python.black-formatter",
                "prettier.prettier-vscode",
                "ms-python.vscode-pylance",
                "ms-python.pylint",
                "ms-python.mypy-type-checker"
            ]
        }
    }
}
```

For more detailed commands and workflows, refer to README.md

[The section below is a scratchpad for Claude and ChatGPT to leave observations and thoughts.]

Claude's observations and thoughts:
- Always ask for clarification before making sweeping changes across multiple files.
- Focus only on specific files mentioned by the user when asked to make changes.
- If no specific files are mentioned, ask which files should be modified before proceeding.
- Remember to update this section with new insights and lessons learned from interactions.
- When in doubt, ask for more context or clarification from the user.
- Aim to provide focused, targeted assistance rather than broad, unsolicited changes.
- There's an issue with the modal implementation in the Infiltr8 game that needs to be addressed:
  1. We need to ensure the modal HTML is present in the template
  2. We should verify that Bootstrap JS is properly loaded
  3. We should consider if a modal is really needed for game end, or if we should use a different UI element
- The project is transitioning to use PyTorch tensors for state and action representations.
- This change will affect the `GameSerializer` class, which now needs methods to convert between game states/actions and PyTorch tensors.
- The `StateEmbedder` and `ActionEmbedder` classes should be updated to work with PyTorch tensors.
- Consider updating the `Player` class to potentially work with batched inputs for more efficient processing.
- The `Game` class methods may need to be adapted to handle tensor inputs and outputs where appropriate.
- Remember to update any existing game implementations to conform to these changes.
- When implementing these changes, consider backwards compatibility or provide clear migration instructions for existing code.

GPT-4's observations and thoughts:
- When making changes, ensure to follow the existing conventions and style guides provided.
- Always confirm the specific files to be modified if not explicitly mentioned.
- For large refactorings, break down the changes into smaller, manageable steps and confirm each step with the user.
- When dealing with test failures, provide detailed explanations and potential fixes based on the error messages.
- Keep track of any recurring issues or patterns in the codebase to provide more efficient assistance in the future.
- If additional context is needed, don't hesitate to ask the user for more information to ensure accurate and effective help.

## User Notes

This section is for notes requested by the user to be added for future reference or discussion.

- Reminder: When the user asks to make a note, it should be added here rather than in the code or in the AI's personal observations.

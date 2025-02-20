# RGI

This repository if a testbed for playing with Reinforcement Learning for Games.

## Goals:
- Reimplement AlphaZero and MuZero (etc.) on a small scale
  - MCTS
  - Policy/Value networks
  - etc.
- Test out different modeling assumptinos and ideas
  - Vanilla transformers
  - Vanilla CNNs
  - LLM with fine tuning?
- Expand the class of games that can be learned
  - Multiplayer
  - Imperfect information
  - "Euro" style games
- Implement multiple board games
- Implement a web UI to play the games
- Implement in multiple frameworks to learn & compare
  - Tensorflow
  - PyTorch
  - Jax


# Usage:

### Build & run
```
docker build -t rgi-gpu .  && \
docker run -it --gpus all -v $(pwd):/rgi-src rgi-gpu /app/scripts/check_rgi_setup.py
```

### Check GPU is working properly in docker image.
```
check_rgi_setup.py
```

### Create a new game from the simple template game.
```
create_game_from_template.sh othello
```

### Build web_app.
```
yarn tsc --watch  # or compile_typescript.sh
```

### Run tests.
```
python web_app/app/main.py   # Start web_app
run_tests.sh --all           # Run all tests
```

### Run linters.
```
run_linter.sh --all
```
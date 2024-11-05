# Build & run
```
docker build -t rgi-gpu .  && \
docker run -it --gpus all -v $(pwd):/rgi-src rgi-gpu /app/scripts/check_rgi_setup.py
```

# Check GPU is working properly in docker image.
```
check_rgi_setup.py
```

# Create a new game from the simple template game.
```
create_game_from_template.sh othello
```

# Build web_app.
```
yarn tsc --watch  # or compile_typescript.sh
```

# Run tests.
```
python web_app/app/main.py   # Start web_app
run_tests.sh --all           # Run all tests
```

# Run linters.
```
run_linter.sh --all
```

## TODO: How much of this needs to be in README.md?

# Launch tensorboard to https://localhost:6006
```
./scripts/start_tensorboard.sh logs
```


# Run 100 games.
```
python main.py --player1 minimax --player2 random --num_games 100
```

# Play vs computer.
```
python rgi/main.py --player1 human --player2 minimax --game connect4
python rgi/main.py --player1 human --player2 minimax --game othello
python rgi/main.py --player1 human --player2 minimax --game infiltr8
```

# Profile code
```
pip install line_profiler
time python -m cProfile -s cumtime rgi/main.py --player1 minimax --player2 random --num_games 5
```

# Create a tarball to upload to LLMs
```
./scripts/create_bot_tarball.sh
```

# Create tarball of source code to make interacting with LLM bots easier.
# Should run from rgi root directory.
find . \( -name "*.py" -o -name "*.sh" -o -name "Dockerfile" -o -name "requirements.txt" -o -path "./bot_artifacts/" \) \
    -not -path "*/.ipynb_checkpoints/*" -not -name "." -type f -print0 | \
xargs -0 -I {} sh -c 'echo -e "\n===== {} =====\n"; cat "{}"' | xclip -selection clipboard



# Web UI
```
pip install fastapi uvicorn[standard]
```
your_project/
├── rgi/                     # Your existing game logic
├── web_app/
│   ├── main.py              # FastAPI application
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── connect4.py
│   │   └── othello.py
│   ├── templates/           # HTML templates
│   │   ├── index.html
│   │   ├── connect4.html
│   │   └── othello.html
│   └── static/              # CSS and JavaScript files
│       ├── styles.css
│       ├── connect4.js
│       └── othello.js
└── ...


# Create training data
```
time python rgi/main.py --player1=random --player2=random --num_games=1000 --save_trajectories
```

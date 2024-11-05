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
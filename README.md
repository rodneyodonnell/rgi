# Build & run
```
docker build -t rgi-gpu .  && \
docker run -it --gpus all -v $(pwd)/logs:/app/logs rgi-gpu
```

# Run using existing image, but updated /rgi
docker run -it --gpus all -v $(pwd)/rgi:/app/rgi -v $(pwd)/logs:/app/logs rgi-gpu


# Launch tensorboard to https://localhost:6006
```
./scripts/start_tensorboard.sh logs
```

# Run tests:
```
pytest
```

# Reformat code:
```
black . --line-length 120
```

# Run 100 games.
```
python main.py --player1 minimax --player2 random --num_games 100
```

# Play vs computer.
```
python rgi/main.py --player1 human --player2 minimax --game connect4
```

# Profile code
```
pip install line_profiler
time python -m cProfile -s cumtime rgi/main.py --player1 minimax --player2 random --num_games 5
```

# Create a tarball to upload to LLMs
```
find . \( -name "*.py" -o -name "Dockerfile" -o -name "requirements.txt" -o -path "./bot_artifacts/*" \) \
    -not -path "*/.ipynb_checkpoints/*" -not -name ".*" | tar -czvf rgi_source.tar.gz -T -

```
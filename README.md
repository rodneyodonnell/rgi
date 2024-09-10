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
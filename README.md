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
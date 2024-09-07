# Build & run
```
docker build -t tensorflow-gpu-test .  && \
docker run -it --gpus all -v $(pwd)/logs:/app/logs tensorflow-gpu-test
```

# Run using existing image, but updated /src
docker run -it --gpus all -v $(pwd)/src:/app/src -v $(pwd)/logs:/app/logs tensorflow-gpu-test


# Launch tensorboard to https://localhost:6006
```
./scripts/start_tensorboard.sh logs
```
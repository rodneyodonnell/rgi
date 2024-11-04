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
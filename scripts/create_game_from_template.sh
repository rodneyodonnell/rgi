#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 game_name"
    echo "Example: $0 tic_tac_toe"
    exit 1
fi

GAME_NAME=$(echo "$1" | tr '[:upper:]' '[:lower:]')
PASCAL_NAME=$(echo "$GAME_NAME" | sed -r 's/(^|_)([a-z])/\U\2/g')

# Setup paths
GAMES_DIR="rgi/games"
TESTS_DIR="rgi/tests/games"

# Check if directories already exist
if [ -d "$GAMES_DIR/$GAME_NAME" ] || [ -d "$TESTS_DIR/$GAME_NAME" ]; then
    echo "Error: Game directories already exist"
    exit 1
fi

# Copy and modify game files
cp -r "$GAMES_DIR/count21" "$GAMES_DIR/$GAME_NAME"
cp -r "$TESTS_DIR/count21" "$TESTS_DIR/$GAME_NAME"

# Replace names in all Python files
find "$GAMES_DIR/$GAME_NAME" "$TESTS_DIR/$GAME_NAME" -type f -name "*.py" | while read -r file; do
    sed -i "s/count21/$GAME_NAME/g" "$file"
    sed -i "s/Count21/$PASCAL_NAME/g" "$file"
    # Only rename if the filename contains count21
    if [[ $(basename "$file") == *"count21"* ]]; then
        mv "$file" "$(echo "$file" | sed "s/count21/$GAME_NAME/")"
    fi
done

echo "Successfully created new game '$GAME_NAME'"
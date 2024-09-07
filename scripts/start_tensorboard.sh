#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 /path/to/your/logs"
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
docker build --progress=plain -t tensorboard -f "$SCRIPT_DIR/tensorboard.Dockerfile" "$SCRIPT_DIR"
docker run -it --rm -p 6006:6006 -v "$1":/logs tensorboard



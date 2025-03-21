#!/bin/bash

# Exit on error
set -e

echo "Running formatters..."

# Run isort to sort imports
echo "Running isort..."
isort .

# Run black using project config from pyproject.toml
echo "Running black..."
black . 
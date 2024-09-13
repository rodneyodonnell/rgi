# Create tarball of source code to make interacting with LLM bots easier.
# Should run from rgi root directory.
find . \( -name "*.py" -o -name "*.sh" -o -name "Dockerfile" -o -name "requirements.txt" -o -path "./bot_artifacts/*" \) \
    -not -path "*/.ipynb_checkpoints/*" -not -name ".*" | tar -czvf rgi_source.tar.gz -T -

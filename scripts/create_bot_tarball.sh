# Create tarball of source code to make interacting with LLM bots easier.
# Should run from rgi root directory.
find . \( -name "*.py" -o -name "*.sh" -o -name "Dockerfile" -o -name "requirements.txt" -o -path "./bot_artifacts/*" \) \
    -not -path "*/.ipynb_checkpoints/*" -not -name ".*" | tar -cvf rgi_source.tar -T -

xclip -selection clipboard < rgi_source.tar

echo Created source tarball 'rgi_source.tar' and copied to clipboard.

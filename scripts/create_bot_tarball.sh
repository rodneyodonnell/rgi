# Create tarball of source code to make interacting with LLM bots easier.
# Should run from rgi root directory.

# If specified, exclude files matching this pattern.
EXCLUDE_PATTERN=${1:-'^$'}

# Find & create tarball from all files we want to add as context for an LLM.
find . \( -path "*/node_modules" -o \
          -path "*/.git" -o \
          -path "*/.pytest_cache" -o \
          -path "*/.ipynb_checkpoints" -o \
          -path "*/pdfs" -o \
          -path "*/logs" -o \
          -path "*/__pycache__" \) -prune \
       -o -not -name "TODO.md" \
       -not -name ".*" \
       -not -name "*.tar" \
       -not -name "package-lock.json" \
       -type f \
       -print | grep -v -E "$EXCLUDE_PATTERN" | tar -cvf rgi_source.tar -T -


xclip -selection clipboard < rgi_source.tar


echo
echo Created source tarball 'rgi_source.tar' and copied to clipboard.
ls -lSh rgi_source.tar
echo
echo "Top 10 largest files in tarball:"
tar -tvf rgi_source.tar | awk '{print $3, $6}' | numfmt --to=iec --suffix=B --padding=7 | sort -rh | head
